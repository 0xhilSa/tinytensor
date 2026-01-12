#!/bin/bash

srcdir=./tinytensor/engine
gcc -Wall -Wextra -g "$srcdir/test.c" "$srcdir/array.c" -o "$srcdir/test"
