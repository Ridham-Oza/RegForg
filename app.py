from ui.gradio_app import (
    create_gradio_app
)


def main():

    app = create_gradio_app()

    app.launch(
        server_name="127.0.0.1",
        server_port=7860
    )


if __name__ == "__main__":
    main()