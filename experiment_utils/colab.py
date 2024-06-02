import logging
from pathlib import Path


def init(local_drives_dir: str = None, account="ahmed.younes.sam@gmail.com") -> Path:
    """
    Initializes the environment by detecting whether it's running in Google Colab or locally.
    Automatically mounts Google Drive if in Colab and uses a default or specified local directory path if running locally.
    Additionally, tries to dynamically find Google Drive paths in common alternative locations if the default is not specified.

    Args:
        default_local_dir (str): Optional default path to the local directory if not in Colab.
        account (str): Email address associated with the Google Drive account to find specific user directory.

    Returns:
        Path: Path to the directory based on the environment.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        ipython = get_ipython()
        if "google.colab" in str(ipython):
            from google.colab import drive

            drive.mount("/content/drive", force_remount=True)
            return Path("/content/drive/My Drive/")
    except NameError:
        logging.info(
            "Not in Google Colab, proceeding with local or specified directory setup."
        )
    except Exception as e:
        logging.error(f"Error during initialization: {e}")
        raise e

    resolved_path = (
        Path(local_drives_dir).expanduser().resolve()
        if local_drives_dir
        else Path.home()
    )

    if resolved_path.exists():
        logging.info(f"Using resolved path: {resolved_path}")
        return resolved_path
    else:
        logging.warning(
            f"Resolved path does not exist, checking alternative paths: {resolved_path}"
        )
        alternative_path = Path("~/Library/CloudStorage").expanduser().resolve()
        if alternative_path.exists():
            for child in alternative_path.iterdir():
                if (
                    "GoogleDrive" in str(child)
                    and child.is_dir()
                    and account in str(child)
                ):
                    logging.info(
                        f"Found Google Drive directory for account {account}: {child}"
                    )
                    return child / local_drives_dir
        logging.error(
            "No valid path found. Ensure that the specified paths are correct."
        )
        return None
