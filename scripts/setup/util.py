import subprocess
import configparser
import os
import shutil
import hashlib
# from sh import wget, cat, grep, sudo, echo, export

config = configparser.ConfigParser(
    interpolation=configparser.ExtendedInterpolation())


def init(configfile):
    config.read(configfile)


class ValidationError(Exception):
    pass


class DownloadError(Exception):
    pass


def download(url, dest):
    """ Tries to download the archive at `url` into `dest` folder with the given `name`.

    `url` remote archive to download

    `dest` local download destination
    """
    result = subprocess.run(
        ['wget', '-c', '--read-timeout=5', '--tries=0', url, '-P', dest])
    if result.returncode != 0:
        raise DownloadError(f"Failed to download: {url}")
    # if subprocess.run(["wget", "-c", "--read-timeout=5",
    #                    "--tries=0", url, "-P", dest]).returncode != 0:


def validate_checksum(file, mode, expected):
    actual = ""
    if mode == "sha256":
        with open(file, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
    elif mode == "sha512":
        with open(file, "rb") as f:
            actual = hashlib.sha512(f.read()).hexdigest()
    elif mode == "skip":
        return
    else:
        raise

    if actual != expected:
        raise ValidationError(f"Checksum failed: {actual} != {expected}")


def unpack(path, dest):
    shutil.unpack_archive(path, extract_dir=dest)


def rm(path):
    os.remove(path)


def check_dir():
    cwd = os.getcwd()
    expected = os.path.join(config["workspace"]["root"], "scripts")
    if cwd != expected:
        raise RuntimeError(
            'Calling script from unexpected directory: expected={0}, got={1}'.format(expected, cwd))


def __check_env(line):
    cat = subprocess.Popen(
        ['cat', config['workspace']['env_file']],
        stdout=subprocess.PIPE)
    grep = subprocess.Popen(['grep', '-q', line], stdin=cat.stdout)
    return grep.returncode == 0


def __add_env(line):
    echo = subprocess.Popen(['echo', line], stdout=subprocess.PIPE)
    tee = subprocess.run(
        ['sudo', 'tee', '-a', config['workspace']['env_file']],
        stdin=echo.stdout, stdout=subprocess.DEVNULL)
    if tee.returncode != 0:
        raise RuntimeError('Failed to add line to environment: ' + line)

# When adding a new path, we first check if the path exists then add it to both the `.bashrc` file and to the `PATH` environment variable.


def try_add_path(path):
    if not __check_env(path):
        __add_env('export PATH=$PATH:' + path)
        os.environ['PATH'] += os.pathsep + path


def try_add_bashrc(line):
    if not __check_env(line):
        __add_env(line)
