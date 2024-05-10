from numba import jit as nb_jit
import typing as tp

__all__ = ["WrapNumba", "jit"]

Kwargs = tp.Dict[str, tp.Any]


class WrapNumba:
    """
    Class for decorating functions using Numba.

    Parameters:
    ----------
    fix_cannot_parallel : bool, optional
        Whether to set `parallel` to False if there is no 'can_parallel' tag.
    nopython : bool, optional
        Whether to run in nopython mode.
    nogil : bool, optional
        Whether to release the GIL (Global Interpreter Lock).
    parallel : bool, optional
        Whether to enable automatic parallelization.
    cache : bool, optional
        Whether to write the result of function compilation into a file-based 
        cache.
    **options
        Additional keyword arguments passed to the Numba decorator.

    Attributes:
    ----------
    fix_cannot_parallel : bool
        Whether to set `parallel` to False if there is no 'can_parallel' tag.
    options : dict
        Options passed to the Numba decorator.
    nopython : bool
        Whether to run in nopython mode.
    nogil : bool
        Whether to release the GIL.
    parallel : bool
        Whether to enable automatic parallelization.
    boundscheck : bool
        Whether to enable bounds checking for array indices.
    cache : bool
        Whether to write the result of function compilation into a file-based 
        cache.
    """

    def __init__(
        self,
        fix_cannot_parallel: bool = True,
        nopython: bool = True,
        nogil: bool = True,
        parallel: bool = False,
        cache: bool = False,
        boundscheck: bool = False,
        **options,
    ) -> None:
        self._fix_cannot_parallel = fix_cannot_parallel
        self._nopython = nopython
        self._nogil = nogil
        self._parallel = parallel
        self._cache = cache
        self._boundscheck = boundscheck
        self._options = options

    @property
    def fix_cannot_parallel(self) -> bool:
        """
        Whether to set `parallel` to False if there is no 'can_parallel' tag.

        Returns:
        -------
        bool
            True if fixing cannot parallel is enabled, False otherwise.
        """
        return self._fix_cannot_parallel

    @property
    def options(self) -> Kwargs:
        """
        Options passed to the Numba decorator.

        Returns:
        -------
        dict
            The options dictionary.
        """
        return self._options

    @property
    def nopython(self) -> bool:
        """
        Whether to run in nopython mode.

        Returns:
        -------
        bool
            True if running in nopython mode, False otherwise.
        """
        return self._nopython

    @property
    def nogil(self) -> bool:
        """
        Whether to release the GIL.

        Returns:
        -------
        bool
            True if releasing the GIL, False otherwise.
        """
        return self._nogil

    @property
    def parallel(self) -> bool:
        """
        Whether to enable automatic parallelization.

        Returns:
        -------
        bool
            True if automatic parallelization is enabled, False otherwise.
        """
        return self._parallel

    @property
    def boundscheck(self) -> bool:
        """
        Whether to enable bounds checking for array indices.

        Returns:
        -------
        bool
            True if bounds checking is enabled, False otherwise.
        """
        return self._boundscheck

    @property
    def cache(self) -> bool:
        """
        Whether to write the result of function compilation into a file-based
        cache.

        Returns:
        -------
        bool
            True if caching is enabled, False otherwise.
        """
        return self._cache

    def decorate(
        self,
        py_func: tp.Callable,
        tags: tp.Optional[set] = None
    ) -> tp.Callable:
        """
        Decorate a Python function using the configured Numba options.

        Parameters:
        ----------
        py_func : Callable
            The Python function to decorate.
        tags : set, optional
            Optional set of tags that influence the decoration.

        Returns:
        -------
        Callable
            The decorated function.
        """
        tags = tags or set()
        options = dict(self.options)
        parallel = self.parallel

        if self.fix_cannot_parallel and parallel and "can_parallel" not in tags:
            parallel = False

        cache = self.cache

        if parallel and cache:
            cache = False

        return nb_jit(
            nopython=self.nopython,
            nogil=self.nogil,
            parallel=parallel,
            cache=cache,
            boundscheck=self.boundscheck,
            **options,
        )(py_func)


def jit(
    py_func: tp.Optional[tp.Callable] = None,
    tags: tp.Optional[set] = None,
    **options,
) -> tp.Callable:
    """
    Decorator factory for JIT (Just-In-Time) compilation using Numba.

    Parameters:
    ----------
    py_func : Callable, optional
        The Python function to decorate.
    tags : set, optional
        Optional set of tags that influence the decoration.
    **options
        Additional keyword arguments passed to the Numba decorator.

    Returns:
    -------
    Callable
        The decorated function.
    """
    def decorator(_py_func: tp.Callable) -> tp.Callable:
        nonlocal options
        model = WrapNumba(**options)
        return model.decorate(_py_func, tags=tags)

    return decorator if py_func is None else decorator(py_func)
