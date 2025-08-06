import os
import sys
import PyInstaller.__main__
import shutil

def build():
    APP_NAME = "Controladores"
    MAIN_SCRIPT = "TCC.py"
    ICON_PATH = "icon2.ico"
    sep = ";" if sys.platform == "win32" else ":"

    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    cmd = [
        MAIN_SCRIPT,
        f"--name={APP_NAME}",
        "--onefile",
        "--windowed",
        f"--icon={ICON_PATH}",
        "--noconfirm",
        "--clean",
        f"--add-data={ICON_PATH}{sep}.",
        "--hidden-import=control.matlab",
        "--hidden-import=scipy.special.cython_special",
        "--hidden-import=matplotlib.backends.backend_tkagg",
        "--hidden-import=PIL._tkinter_finder",
        "--exclude-module=matplotlib.tests",
        "--exclude-module=numpy.random._examples",
    ]

    print("\nComando que será executado:")
    print(" ".join(cmd))

    PyInstaller.__main__.run(cmd)
    print(f"\n✅ Executável criado em: dist/{APP_NAME}.exe")

if __name__ == "__main__":
    build()
