# how to produce the wrapped algo609 lib
FOPTS='-ffixed-line-length-72 -std=legacy'
#  -Wextra -Wall -fPIC
ONLYS='dbskin bskin'

# Daniele uses gfortran from gcc 8.3.0
# GCCPATH=/data/tmplca/dtomatis/tools/gcc/gcc-8.3.0
F77=gfortran
F90=gfortran
# INC=$GCCPATH/lib64
# INC=/c/msys64/mingw64/x86_64-w64-mingw32/lib/
INC=/c/msys64/mingw64/lib/gcc/x86_64-w64-mingw32/9.3.0/
#INC=/c/msys64/mingw64/x86_64-w64-mingw32/lib/
#PYT=c:/Users/dt247226/AppData/Local/Programs/Python/Python38/python.exe
PYT=python
LIBS=-llibquadmath

unset LD_LIBRARY_FLAGS   # just in case was set before
export LDFLAGS=-Wl,-rpath=$INC
export NPY_DISTUTILS_APPEND_FLAGS=1
#  --f77exec=$F77 --f90exec=$F90
$PYT -m numpy.f2py -c --f77flags="$FOPTS" --f90flags="$FOPTS" \
  --fcompiler=gnu95 --compiler=mingw32 algo609.f \
  only: $ONLYS : -m algo609
