import logging
import sys
from pathlib import Path
import sys
import subprocess
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def init(drive_folder: str = 'My Drive', drive_mount: str = 'drive', account="ahmed.younes.sam@gmail.com") -> Path:
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

    try:
        ipython = get_ipython()
        if "google.colab" in str(ipython):
            from google.colab import drive

            setup_logging()

            drive.mount(f"/content/{drive_mount}", force_remount=True)
            return Path(f"/content/{drive_mount}/{drive_folder}/")
    except NameError:
        logging.info(
            "Not in Google Colab, proceeding with local or specified directory setup."
        )
    except Exception as e:
        logging.error("Error during initialization: %s", e)
        raise e

    resolved_path = (
        Path(drive_folder).expanduser().resolve()
        if drive_folder
        else Path.home()
    )

    if resolved_path.exists():
        logging.info("Using resolved path: %s", resolved_path)
        return resolved_path
    else:
        logging.warning(
            "Resolved path does not exist, checking alternative paths: %s",
            resolved_path,
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
                        "Found Google Drive directory for account %s: %s",
                        account,
                        child,
                    )
                    return child / drive_folder
        logging.error(
            "No valid path found. Ensure that the specified paths are correct."
        )
        return None


def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    if not logger.handlers:  # To ensure no duplicate handlers are added
        # Create handler that logs to sys.stdout (standard output)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)  # Adjust the logging level as needed

        # Create formatter and add it to the handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger



def save_environment_info(base_path: Path):
    """
    Saves the Python version and installed packages to files in the specified directory.

    Args:
        base_path (Path): The base directory where the files will be saved.
    """
    try:
        # Get Python version
        python_version = sys.version.split(" ")[0]  # Just the version number

        # Get installed packages using pip freeze
        pip_freeze_output = subprocess.check_output(["pip", "freeze"]).decode("utf-8")

        # Define file paths
        python_version_file = base_path / "python_version.txt"
        requirements_file = base_path / "requirements.txt"

        # Save Python version to a file
        with open(python_version_file, "w") as file:
            file.write(f"python=={python_version}\n")

        # Save the installed packages to a requirements file
        with open(requirements_file, "w") as file:
            file.write(pip_freeze_output)

        logging.info(f"Python version saved to {python_version_file}")
        logging.info(f"Installed packages saved to {requirements_file}")

    except Exception as e:
        logging.error(f"Error saving environment info: {e}")
        raise
