"""Private Windows DOC -> DOCX worker, invoked by document_reader only.

Exit codes:
    2: Missing dependencies.
    3: Word unavailable.
    4: Document conversion failure.
    5: Process isolation unavailable.

Never use the user's existing Word instance.
"""

import sys
from typing import cast
from uuid import uuid4


def convert(source: str, target: str) -> int:
    try:
        import pythoncom
        import win32api
        import win32com.client
        import win32con
        import win32job
        import win32process
    except ImportError:
        return 2

    pythoncom.CoInitialize()

    word = None
    document = None
    job: int | None = None
    owned = False

    try:
        existing_pids = set(win32process.EnumProcesses())

        try:
            word = win32com.client.DispatchEx("Word.Application")
        except Exception:
            return 3

        try:
            _, pid = win32process.GetWindowThreadProcessId(
                word.Hwnd
            )

            if pid in existing_pids:
                # Do not change or close a pre-existing Word process.
                return 5

            owned = True

            # Some pywin32 stubs incorrectly describe this return
            # value as None. At runtime it is a native handle.
            job = cast(
                int,
                win32job.CreateJobObject(None, ""),
            )

            info = win32job.QueryInformationJobObject(
                job,
                win32job.JobObjectExtendedLimitInformation,
            )

            info["BasicLimitInformation"]["LimitFlags"] |= (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )

            win32job.SetInformationJobObject(
                job,
                win32job.JobObjectExtendedLimitInformation,
                info,
            )

            process = win32api.OpenProcess(
                win32con.PROCESS_SET_QUOTA
                | win32con.PROCESS_TERMINATE,
                False,
                pid,
            )

            try:
                win32job.AssignProcessToJobObject(
                    job,
                    process,
                )
            finally:
                win32api.CloseHandle(process)

        except Exception:
            return 5

        word.Visible = False
        word.DisplayAlerts = 0

        # msoAutomationSecurityForceDisable
        word.AutomationSecurity = 3
        word.Options.UpdateLinksAtOpen = False

        # Protected documents should fail rather than waiting for
        # an invisible password dialog. No password is saved.
        password = uuid4().hex

        document = word.Documents.Open(
            FileName=source,
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
            Visible=False,
            NoEncodingDialog=True,
            PasswordDocument=password,
            PasswordTemplate=password,
            WritePasswordDocument=password,
            WritePasswordTemplate=password,
        )

        # 12 = wdFormatXMLDocument, a non-macro DOCX document.
        document.SaveAs2(
            FileName=target,
            FileFormat=12,
            AddToRecentFiles=False,
        )

        return 0

    except Exception:
        return 4

    finally:
        if owned:
            if document is not None:
                try:
                    document.Close(SaveChanges=0)
                except Exception:
                    pass

            if word is not None:
                try:
                    word.Quit(SaveChanges=0)
                except Exception:
                    pass

        # Closing the job terminates only this worker's Word process.
        # Windows also closes the handle if the worker is terminated
        # by the parent's timeout.
        if job is not None:
            win32api.CloseHandle(job)

        document = None
        word = None

        pythoncom.CoUninitialize()


if __name__ == "__main__":
    raise SystemExit(
        convert(sys.argv[1], sys.argv[2])
        if len(sys.argv) == 3
        else 4
    )