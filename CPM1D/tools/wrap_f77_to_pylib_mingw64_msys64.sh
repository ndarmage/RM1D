# how to produce the wrapped algo609 lib
LIBS='-lquadmath'
FOPTS='-ffixed-line-length-72 -std=legacy'
#  -Wextra -Wall -fPIC
ONLYS='dbskin bskin'

GCCPATH=/c/msys64/mingw64/bin/
F77=$GCCPATH/gfortran
F90=$GCCPATH/gfortran
# INC=$GCCPATH/lib64
# INC=/c/msys64/mingw64/x86_64-w64-mingw32/lib/
#INC=/c/msys64/mingw64/lib/gcc/x86_64-w64-mingw32/10.2.0/
INC=/c/msys64/mingw64/bin
#INC=/c/msys64/mingw64/x86_64-w64-mingw32/lib/
PYT=/c/Users/admin-local/AppData/Local/Programs/Python/Python38/python.exe
#PYT=python

unset LD_LIBRARY_FLAGS   # just in case was set before
export LDFLAGS=-Wl,-rpath=$INC
export NPY_DISTUTILS_APPEND_FLAGS=1
#  --f77exec=$F77 --f90exec=$F90
$PYT -m numpy.f2py --verbose -c --f77flags="$FOPTS" --f90flags="$FOPTS" \
  --fcompiler=gnu95 --compiler=mingw32 algo609.f \
  only: $ONLYS : -L/c/msys64/mingw64 -L$INC $LIBS -m algo609
