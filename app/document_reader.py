"""Read uploads into the TXT/JSON contract accepted by transcript_parser.

DOCX is read locally. Legacy DOC is converted by an isolated Windows Word
worker. No document text is uploaded to a conversion service or globally cached.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UNPACKED_BYTES = 80 * 1024 * 1024
SUPPORTED_EXTENSIONS = ("txt", "json", "doc", "docx")


class DocumentReadError(ValueError):
    """A document could not be read; the message is suitable for the UI."""


@dataclass(frozen=True)
class DocumentInput:
    text: str
    parser_format: str
    source_format: str
    warnings: tuple[str, ...] = ()


def _label(text: str) -> str:
    return " ".join(
        text.translate(str.maketrans("Iİ", "ıi")).lower().split()
    ).strip(":")


def _table_lines(table, warnings: list[str]) -> list[str]:
    """Map explicit speaker/text headers; preserve other tables for review."""
    rows = list(table.rows)

    if not rows:
        return []

    first = [_label(cell.text) for cell in rows[0].cells]

    speaker_headers = {"konuşmacı", "speaker", "kişi", "rol"}
    text_headers = {"metin", "konuşma", "ifade", "text", "utterance"}

    speaker_columns = [
        index
        for index, value in enumerate(first)
        if value in speaker_headers
    ]

    text_columns = [
        index
        for index, value in enumerate(first)
        if value in text_headers
    ]

    # Only an explicit two-column schema is transformed automatically.
    # Extra columns remain visible rather than being discarded.
    if (
        len(first) == 2
        and len(speaker_columns) == len(text_columns) == 1
        and speaker_columns[0] != text_columns[0]
    ):
        speaker_column = speaker_columns[0]
        text_column = text_columns[0]

        # Merged cells and nested tables require review.
        if all(
            len(row.cells) == 2
            and row.cells[0]._tc is not row.cells[1]._tc
            and not any(cell.tables for cell in row.cells)
            and (
                not row.cells[text_column].text.strip()
                or row.cells[speaker_column].text.strip().rstrip(":")
            )
            for row in rows
        ):
            lines = []
            seen_text_cells = set()

            for row in rows[1:]:
                cell = row.cells[text_column]

                if cell._tc in seen_text_cells:
                    continue

                seen_text_cells.add(cell._tc)

                name = " ".join(
                    row.cells[speaker_column].text.split()
                ).rstrip(":")

                speech = cell.text.strip()

                if speech:
                    lines.append(f"{name}: {speech}")

            return lines

    warnings.append(
        "Bir tablo konuşmacı/metin tablosu olarak tanınmadı. "
        "Hücreler sırayla korundu; önizlemede konuşmacı etiketlerini "
        "ve tablo içeriğini kontrol edin."
    )

    lines = []
    seen_cells = set()

    for row in rows:
        for cell in row.cells:
            if cell._tc in seen_cells:
                continue

            seen_cells.add(cell._tc)
            lines.extend(_block_lines(cell, warnings))

    return lines


def _block_lines(container, warnings: list[str]) -> list[str]:
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    lines = []

    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            if block.text.strip():
                lines.append(block.text.strip())

        elif isinstance(block, Table):
            lines.extend(_table_lines(block, warnings))

    return lines


def _read_docx(data: bytes) -> tuple[str, tuple[str, ...]]:
    # Validate the package before passing it to python-docx.
    try:
        with ZipFile(BytesIO(data)) as archive:
            if "word/document.xml" not in archive.namelist():
                raise DocumentReadError(
                    "Dosya geçerli bir DOCX belgesi değil."
                )

            unpacked_size = sum(
                entry.file_size for entry in archive.infolist()
            )

            if unpacked_size > MAX_UNPACKED_BYTES:
                raise DocumentReadError(
                    "Word belgesinin açılmış içeriği çok büyük "
                    "(80 MB sınırı)."
                )

            if any(
                entry.flag_bits & 1
                for entry in archive.infolist()
            ):
                raise DocumentReadError(
                    "Şifreli Word dosyalarını şifresiz bir kopya "
                    "olarak kaydedin."
                )

    except BadZipFile as error:
        raise DocumentReadError(
            "DOCX paketi okunamadı. Dosya bozuk, şifreli "
            "veya uzantısı yanlış olabilir."
        ) from error

    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
        from lxml.etree import XMLSyntaxError
    except ImportError as error:
        raise DocumentReadError(
            "DOCX okuyucusu eksik. Proje klasöründe "
            "py -m pip install -r requirements.txt çalıştırın."
        ) from error

    warnings = [
        "Word belgesinin ana gövdesi okunur. Başlıkları ve senaryo "
        "notlarını önizlemede çıkarın; üst/alt bilgiler, dipnotlar "
        "ve resimlerdeki yazılar analize dahil edilmez."
    ]

    try:
        document = Document(BytesIO(data))

        if document.element.xpath(
            ".//w:ins | .//w:del | .//w:txbxContent | .//w:sdt"
        ):
            warnings.append(
                "Belgede değişiklik izleri, metin kutuları veya "
                "içerik denetimleri var. Bu alanların metni eksik "
                "olabilir; Word'de sadeleştirip yeniden yükleyin."
            )

        lines = _block_lines(document, warnings)

    except (
        BadZipFile,
        PackageNotFoundError,
        XMLSyntaxError,
        KeyError,
        ValueError,
    ) as error:
        raise DocumentReadError(
            "Word belgesinin yapısı okunamadı; "
            "Word'den yeniden DOCX olarak kaydedin."
        ) from error

    except AttributeError as error:
        raise DocumentReadError(
            "Word okuyucu sürümü uyumsuz. "
            "py -m pip install -r requirements.txt ile güncelleyin."
        ) from error

    text = "\n".join(lines)

    if not text.strip():
        raise DocumentReadError(
            "Belgede okunabilir ana gövde metni bulunamadı. "
            "Taranmış sayfalar için OCR gerekir."
        )

    return text, tuple(dict.fromkeys(warnings))


def _convert_doc(data: bytes) -> bytes:
    if sys.platform != "win32":
        raise DocumentReadError(
            "DOC dönüştürme bu sürümde Windows ve masaüstü "
            "Microsoft Word gerektirir. Bu ortamda dosyayı "
            "DOCX olarak kaydedip yükleyin."
        )

    is_binary_doc = data.startswith(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    )
    is_rtf = data.lstrip().startswith(b"{\\rtf")

    if not (is_binary_doc or is_rtf):
        raise DocumentReadError(
            "Dosya desteklenen bir DOC/RTF belgesi değil; "
            "Word'den DOCX olarak kaydedin."
        )

    worker = Path(__file__).with_name("word_converter.py")

    with TemporaryDirectory(prefix="medisysai-word-") as folder:
        source = Path(folder) / "input.doc"
        target = Path(folder) / "output.docx"

        source.write_bytes(data)

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(worker),
                    str(source),
                    str(target),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=False,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0,
                ),
            )

        except subprocess.TimeoutExpired as error:
            raise DocumentReadError(
                "Word dönüşümü 60 saniyede tamamlanamadı. "
                "Dosyayı Word'de açıp şifresiz DOCX olarak "
                "kaydetmeyi deneyin."
            ) from error

        except OSError as error:
            raise DocumentReadError(
                "Word dönüştürücü başlatılamadı."
            ) from error

        errors = {
            2: (
                "DOC desteği için pywin32 eksik. "
                "py -m pip install -r requirements.txt çalıştırın."
            ),
            3: (
                "Microsoft Word başlatılamadı. Masaüstü Word'ün "
                "kurulu ve açılabilir olması gerekiyor."
            ),
            4: (
                "DOC dosyası dönüştürülemedi. Dosya şifreli, "
                "bozuk veya Word tarafından engellenmiş olabilir."
            ),
            5: (
                "Ayrı bir Word dönüştürme işlemi oluşturulamadı. "
                "Dosyayı Word'den DOCX olarak kaydedin."
            ),
        }

        if completed.returncode != 0 or not target.is_file():
            raise DocumentReadError(
                errors.get(
                    completed.returncode,
                    "Word dönüşümü tamamlanamadı.",
                )
            )

        if target.stat().st_size > MAX_UPLOAD_BYTES:
            raise DocumentReadError(
                "Dönüştürülen DOCX dosyası 20 MB sınırını aşıyor."
            )

        return target.read_bytes()


def read_document(filename: str, data: bytes) -> DocumentInput:
    extension = Path(filename).suffix.lower().lstrip(".")

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentReadError(
            "Desteklenen dosyalar: TXT, JSON, DOC ve DOCX."
        )

    if not data:
        raise DocumentReadError("Yüklenen dosya boş.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentReadError("Dosya en fazla 20 MB olabilir.")

    if extension in {"txt", "json"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise DocumentReadError(
                "TXT/JSON dosyasını UTF-8 olarak kaydedip "
                "yeniden yükleyin."
            ) from error

        if not text.strip():
            raise DocumentReadError(
                "Dosyada analiz edilecek metin yok."
            )

        return DocumentInput(
            text=text,
            parser_format=extension,
            source_format=extension,
        )

    content = _convert_doc(data) if extension == "doc" else data
    text, warnings = _read_docx(content)

    return DocumentInput(
        text=text,
        parser_format="txt",
        source_format=extension,
        warnings=warnings,
    )