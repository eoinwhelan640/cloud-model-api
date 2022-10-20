# When looking at the log of the webserver, we're initially not logging our
# message "Health request received". This is because we have uvicorn which uses a standard logger
# overriding the python one. We need to add our own logger in tandem so they can coexist.
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",

        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    # This is the name of our custom python logger we'll tell the uvicorn webserver to use.
    "loggers": {
        "ml-ops": {"handlers": ["default"], "level": "INFO"},
    },
}