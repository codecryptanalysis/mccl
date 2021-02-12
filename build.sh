#!/bin/bash

function docmd
{
echo -n "$*..."
if $* ; then
	echo "ok"
else
	echo "failed"
	exit 1
fi
}

docmd autoreconf --install
docmd ./configure
docmd make clean
docmd make all
docmd make check
