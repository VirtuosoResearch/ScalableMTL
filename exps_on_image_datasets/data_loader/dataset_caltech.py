from PIL import Image
import os
import os.path
from typing import Any, Callable, List, Optional, Union, Tuple

from torchvision.datasets.caltech import VisionDataset
from torchvision.datasets.caltech import download_and_extract_archive, verify_str_arg


class Caltech101(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
            self,
            root: str,
            target_type: Union[List[str], str] = "category",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(Caltech101, self).__init__(os.path.join(root, 'caltech101'),
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation"))
                            for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.labels = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.labels.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(os.path.join(self.root,
                                      "101_ObjectCategories",
                                      self.categories[self.labels[index]],
                                      "image_{:04d}.jpg".format(self.index[index])))

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.labels[index])
            elif t == "annotation":
                data = scipy.io.loadmat(os.path.join(self.root,
                                                     "Annotations",
                                                     self.annotation_categories[self.labels[index]],
                                                     "annotation_{:04d}.mat".format(self.index[index])))
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9")
        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
            self.root,
            filename="101_Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91")

    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)


class Caltech256(VisionDataset):
    """`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            idx_start = 0,
            img_num = 60
    ) -> None:
        super(Caltech256, self).__init__(root,
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index: List[int] = []
        self.labels = []
        for (i, c) in enumerate(self.categories):
            files = os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
            for file in files:
                if "jpg" not in file:
                    files.remove(file)
            if idx_start+img_num >= len(files):
                files = files[idx_start:]
            else:
                files = files[idx_start:idx_start + img_num]
            n = len(files)
            self.index.extend(files)
            self.labels.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(os.path.join(self.root,
                                      "256_ObjectCategories",
                                      self.categories[self.labels[index]],
                                      self.index[index])).convert('RGB')

        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            # print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d")