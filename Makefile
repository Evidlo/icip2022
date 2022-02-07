.ONESHELL:

devserver:
	while :;
	do
		inotifywait -e modify *
		tectonic ICIP.tex
	done
