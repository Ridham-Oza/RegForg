"""RegForg entry point:  python app.py  ->  http://127.0.0.1:7860"""

from config import DOCUMENTS_DIR, SERVER_HOST, SERVER_PORT
from src.dashboard.app import CSS, THEME, build_app


def main():
    build_app().queue().launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        theme=THEME,
        css=CSS,
        allowed_paths=[str(DOCUMENTS_DIR.resolve())],  # lets source links open the PDF at the cited page
        show_error=True,
    )


if __name__ == "__main__":
    main()
