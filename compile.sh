#!/bin/bash

srcdir=./tinytensor/engine
gcc -Wall -Wextra -g "$srcdir/test.c" "$srcdir/ten.c" -o "$srcdir/test"
