#!/bin/bash

texfile=ICIP.tex

# start PDF reader in background
evince $(basename ${texfile} .tex).pdf &

# watch for file changes and rebuild
while :;
do
	inotifywait -e create,modify -r *
	beep -f 440 -l 100
	tectonic ICIP.tex
	status=$?
	if [ ${status} -ne 0 ]
	then
		beep -f 440 -r 2
	else
		beep -f 880
	fi
done
