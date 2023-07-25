import logging
import celery
import celery.signals
import celery.utils.log
import urllib.error
import requests.exceptions
from app.lib.duplicate_image_detector import DuplicateImageDetector
from app.lib.google_photos_client import GooglePhotosClient
from app import server  # required for building URLs
from app import CELERY_APP as celery_app
from typing import Callable


class TaskUpdaterLogHandler(logging.Handler):
    """
    Custom logging handler that updates celery task meta
    """

    def __init__(self):
        super().__init__()
        self.update_status = None

    def set_status_updater(self, update_status: Callable[[str], None]):
        self.update_status = update_status

    def emit(self, record):
        # print(f"TaskUpdaterLogHandler: {record.getMessage()}")
        if self.update_status:
            self.update_status(record.getMessage())


task_updater_log_handler = TaskUpdaterLogHandler()

# Update celery task meta with logs from task logger
task_logger = celery.utils.log.get_task_logger(__name__)
task_logger.addHandler(task_updater_log_handler)


# TODO: Can't get this to work, so setting up with a global flag when the task runs instead :(
# Note: after_setup_logger and  signals are called BEFORE
#       stdout is redirected, so we need to listen to a later
# @celery.signals.worker_ready.connect
# def setup_stdout_handler(**kwargs):
#     """
#     Sets up logging handlers to update task metadata on stdout output, so
#     we can pass along progress from the tdqm progress bars from
#     sentence_transformers and our DuplicateImageDetector.
#     """
#     print(f"setup_stdout_handler, kwargs: {kwargs}")
#     # Update celery task meta with logs from redirected output (e.g. print statements)
#     logging.getLogger("celery.redirected").addHandler(task_updater_log_handler)

is_stdout_handler_setup = False

# import torch

# torch.set_num_threads(1)


@celery.shared_task(bind=True)
def process_duplicates(
    self: celery.Task,
    credentials: dict,
    refresh_media_items: bool = False,
):
    def update_status(message, state="PROGRESS"):
        # `meta` comes through as `info` field on task result
        self.update_state(state=state, meta=message)

    task_updater_log_handler.set_status_updater(update_status)

    global is_stdout_handler_setup
    if not is_stdout_handler_setup:
        logging.getLogger("celery.redirected").addHandler(task_updater_log_handler)
        is_stdout_handler_setup = True

    task_logger = celery.utils.log.get_task_logger(__name__)

    try:
        client = GooglePhotosClient(credentials, logger=task_logger)

        if refresh_media_items or client.local_media_items_count() == 0:
            client.fetch_media_items()

        media_items_count = client.local_media_items_count()
        logging.info(f"Processing duplicates for {media_items_count:,} media items...")

        # media_item_groups = list(self.repo.get_media_item_groups())
        # num_groups = len(media_item_groups)
        # num_duplicates = sum([len(group["all"]) for group in media_item_groups])

        # for group in media_item_groups:
        #     for line in pprint.pformat(group).splitlines():
        #         update_status(line)

        # album = __find_or_create_album()

        # update_status(pprint.pformat(album))

        media_items = []
        media_items = client.get_local_media_items()
        duplicate_detector = DuplicateImageDetector(logger=task_logger)
        clusters = duplicate_detector.calculate_clusters(media_items)

        result = {
            "groups": [],
        }

        for group_index, media_item_indices in enumerate(clusters):
            raw_media_items = [media_items[i] for i in media_item_indices]

            # These are already sorted by creationDate asc, so the original mediaItem is the lowest index
            original_media_item_id = media_items[min(media_item_indices)]["id"]

            group_media_items = []
            group = {
                "id": group_index,
                "media_items": group_media_items,
            }

            for raw_media_item in raw_media_items:
                # Remove _id as it's an ObjectId and is not JSON-serializable
                media_item = {
                    k: raw_media_item[k] for k in raw_media_item if k != "_id"
                }
                if raw_media_item["id"] == original_media_item_id:
                    media_item["type"] = "original"
                else:
                    media_item["type"] = "duplicate"
                group_media_items.append(media_item)

            result["groups"].append(group)

        return result

    except urllib.error.HTTPError as error:
        if error.status == 403:
            raise UserFacingError(
                "HTTP Error 403 Forbidden encoutered. Please refresh media items."
            ) from error
        else:
            raise error
    except requests.exceptions.HTTPError as error:
        if error.response.status_code == 401:
            raise UserFacingError(
                "HTTP Error 401 Unauthorized encountered. Please log out and reauthenticate."
            ) from error
        else:
            raise error


class UserFacingError(Exception):
    pass
