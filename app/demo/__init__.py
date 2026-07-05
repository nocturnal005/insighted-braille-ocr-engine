"""Local preview and human demo tools (Stage 3D-H1).

Everything in this package is for local human preview only: it runs the
existing OCR pipeline unchanged and displays the draft result to the person
sitting at the machine. Nothing here alters OCR logic, the /ocr contract,
or logging policy, and nothing here stores images or transcriptions.

All displayed output is an unverified draft and requires QTVI or
Braille-literate specialist verification before any use in teacher
feedback or export.
"""
