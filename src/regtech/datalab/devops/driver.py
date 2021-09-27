# Global import
import os
import tempfile
import shutil

# Local import


class Driver(object):

    def __init__(self, name, desc):
        self.name, self.desc = name, desc

    def __str__(self):
        return '{} - {}'.format(self.name, self.desc)

    def read_file(self, url, **kwargs):
        pass

    def write_file(self, x,  url, **kwargs):
        pass

    def TempFile(self, prefix='tmp', suffix='', dir=None, destination=None, create=False):
        """
        See data_pm.core.drivers.base_driver.TempFile

        :return: TempFile
        """
        return TempFile(driver=self, prefix=prefix, suffix=suffix, dir=dir, destination=destination, create=create)

    def TempDir(self, prefix='tmp', suffix='', dir=None, destination=None, create=False):
        """
        See data_pm.core.drivers.base_driver.TempDir

        :return: TempFile
        """
        return TempDir(driver=self, prefix=prefix, suffix=suffix, dir=dir, destination=destination, create=create)

    def abspath(self, path):
        """
        Returns an absolute path (see os.path.abspath)

        :param path:
        :return: str
        """
        pass

    def basename(self, path):
        """
        Returns the final component of a pathname (see os.path.basename)

        :param str path:
        :return: str
        """
        pass

    def dirname(self, path):
        """
        Returns the directory component of a pathname (see os.path.dirname)

        :param str path:
        :rtype: str
        """
        pass

    def exists(self, path):
        """
        Returns True if the path exists

        :param str path:
        :return: bool
        """
        pass

    def isdir(self, path):
        """
        Returns True if the path is a directory
        Returns False if the path is a file or if it does not exist

        :param str path:
        :return: bool
        """
        pass

    def isfile(self, path):
        """
        Returns True if the path is a file
        Returns False if the path is a directory or if it does not exist

        :param str path:
        :return: bool
        """
        pass

    def join(self, arg, *args):
        """
        Joins two or more pathname components (see os.path.join)

        :param str arg: path start
        :param str args: other path parts
        :return: str
        """
        pass

    def listdir(self, path):
        """
        Lists the content of a directory

        :param str path:
        :return: list[str]
        """
        pass

    def remove(self, path, recursive=False):
        """
        Deletes (optionally recursively) a file or folder

        Will raise an error if the path does not exist

        :param str path:
        :param bool recursive: When True, allow deleting a directory that contains files or subfolders
        """
        pass

    def makedirs(self, path):
        """
        Creates a directory and intermediate directories if they do not exist (like "mkdir -p")

        Won't raise if the path already exists.

        :param str path:
        """
        pass


class FileDriver(Driver):
    tmp_dir = "/tmp"

    def __init__(self, name, desc):
        Driver.__init__(self, name, desc)

    def TempFile(self, prefix='tmp', suffix='', dir=None, destination=None, create=False):
        return super(FileDriver, self).TempFile(
            prefix=prefix,
            suffix=suffix,
            dir=self.tmp_dir,
            destination=destination,
            create=create
        )

    def TempDir(self, prefix='tmp', suffix='', dir=None, destination=None, create=False):
        return super(FileDriver, self).TempDir(
            prefix=prefix,
            suffix=suffix,
            dir=self.tmp_dir,
            destination=destination,
            create=create
        )

    def abspath(self, path):
        return os.path.abspath(path)

    def basename(self, path):
        return os.path.basename(path)

    def dirname(self, path):
        return os.path.dirname(path)

    def exists(self, path):
        return os.path.exists(path)

    def isdir(self, path):
        return os.path.isdir(path)

    def isfile(self, path):
        return os.path.isfile(path)

    def join(self, arg, *args):
        return os.path.join(arg, *args)

    def listdir(self, path):
        return os.listdir(path)

    def remove(self, path, recursive=False):
        if self.isfile(path):
            os.remove(path)
        else:
            if recursive:
                shutil.rmtree(path)
            else:
                os.rmdir(path)

    def makedirs(self, path):
        if not self.exists(path):
            os.makedirs(path)

    def write(self, path):
        return open(path, 'w')

    def read(self, path):
        return open(path, 'r')


class HdhfsDriver(Driver):

    def __init__(self, name, desc):
        Driver.__init__(self, name, desc)


class TempFile(object):
    def __init__(self, driver, prefix='tmp', suffix='', dir=None, destination=None, create=False):
        """
        Temporary file

        The file is automatically deleted when this object is garbage collected.
        If possible, delete the file manually by calling remove().

        Example usage:
            ```
            mytmp = TempFile(driver)
            with driver.write(mytmp.path) as writer:
                writer.write('hello world')

            # Do whatever your want with the file (like uploading it somewhere)

            mytmp.remove()
            ```

        :param BaseDriver driver:
        :param str prefix:
        :param str suffix:
        :param str dir: Where to create the temporary file
        :param str destination: The optional destination of the content of this folder
        """
        self.destination = destination
        self.driver = driver
        self.path = self.driver.abspath(tempfile.mktemp(suffix=suffix, prefix=prefix, dir=dir))

        # Create an empty file
        if create:
            with self.driver.write(self.path) as handle:
                pass

    def move_to_destination(self):
        """
        Move the temporary file to its destination.
        Also works when `self.path` is a directory (hard sync).

        If the destination already exist, it will be deleted before the move.
        """
        if self.path is None:
            raise ValueError('Temporary path has already been moved to destination "{}"'.format(self.destination))

        if self.driver.exists(self.destination):
            self.driver.remove(self.destination, recursive=True)

        self.driver.renames(self.path, self.destination)
        self.remove()

    def remove(self):
        """
        Remove the temporary file/directory

        Once called, self.path is set to None
        """
        if self.path is not None and self.driver.exists(self.path):
            self.driver.remove(self.path, recursive=True)

        self.path = None

    def __del__(self):
        self.remove()


class TempDir(TempFile):
    def __init__(self, driver, prefix='tmp', suffix='', dir=None, destination=None, create=False):
        """
        Temporary directory

        The directory is automatically deleted when this object is garbage collected.
        If possible, delete the directory manually by calling remove().

        Example usage:
            ```
            mytmp = TempDir(driver, destination='final/dest', create=True)
            with driver.write(driver.join(mytmp.path, 'file.txt')) as writer:
                writer.write('hello world')
            mytmp.move_content_to_destination()

            # final/dest now contains a file named file.txt
            ```

        :param BaseDriver driver:
        :param str prefix:
        :param str suffix:
        :param str dir: Where to create the temporary file
        :param str destination: The optional destination of the content of this folder
        """
        # Initiate as a TempFile, without creating the file
        super(self.__class__, self).__init__(
            driver=driver,
            prefix=prefix,
            suffix=suffix,
            dir=dir,
            destination=destination,
            create=False)
        if create:
            self.driver.makedirs(self.path)

    def move_content_to_destination(self):
        """
        Move the content of the temporary directory to its destination (soft sync).

        The content already existing in the destination will be deleted before the move.
        """
        # If destination already exists, copy each file contained in self.path to self.destination
        #  Else, move the whole directory self.path
        if self.driver.exists(self.destination):
            files = self.driver.listdir(self.path)
            for f in files:
                src_path = self.driver.join(self.path, f)
                dst_path = self.driver.join(self.destination, f)

                if self.driver.exists(dst_path):
                    self.driver.remove(dst_path, recursive=True)
                self.driver.renames(src_path, dst_path)

            self.remove()
        else:
            self.move_to_destination()


