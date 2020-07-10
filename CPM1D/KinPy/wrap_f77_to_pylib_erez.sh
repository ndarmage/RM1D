# how to produce the wrapped algo609 lib
FOPTS='-ffixed-line-length-72 -Wextra -Wall -std=legacy'
ONLYS='dbskin bskin'

# Daniele uses gfortran from gcc 8.3.0
#GCCPATH=/data/tmplca/dtomatis/tools/gcc/gcc-8.3.0
#F77=$GCCPATH/bin/gfortran
#INC=$GCCPATH/lib64

# Erez gcc
#GCCPATH=/usr/bin
#F77=$GCCPATH/gfortran
F77=gfortran
#INC=/lib64

unset LD_LIBRARY_FLAGS   # just in case was set before
#export LDFLAGS=-Wl,-rpath=$INC
export NPY_DISTUTILS_APPEND_FLAGS=1
python3 -m numpy.f2py -c --f77flags="$FOPTS" --f90flags="$FOPTS" --f77exec=$F77 --f90exec=$F77 algo609.f only: $ONLYS : -m algo609
