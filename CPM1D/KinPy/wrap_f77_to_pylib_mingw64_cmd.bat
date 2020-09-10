Rem how to produce the wrapped algo609 lib
@echo off
set FOPTS="-ffixed-line-length-72 -std=legacy"
::  -Wextra -Wall -fPIC
set ONLYS=dbskin bskin

:: set GCCPATH=/c/msys64/mingw64
set F77=gfortran
set F90=gfortran
:: set INC=/c/msys64/mingw64/lib/gcc/x86_64-w64-mingw32/10.2.0/
set INC=/c/Strawberry/c/lib/gcc/x86_64-w64-mingw32/8.3.0/
:: set WINC=/c/Windows/SysWOW64
:: INC=/c/msys64/mingw64/x86_64-w64-mingw32/lib/
set PYT=python
set LIBS="-lquadmath"

set LD_LIBRARY_FLAGS=
set LDFLAGS=-Wl,-rpath=%INC%
set NPY_DISTUTILS_APPEND_FLAGS=1
::  --f77exec=$F77 --f90exec=$F90
%PYT% -m numpy.f2py --verbose -c --f77flags=%FOPTS% --f90flags=%FOPTS% ^
  --fcompiler=gnu95 --compiler=mingw32 algo609.f ^
  only: %ONLYS% : %LIBS% -m algo609
