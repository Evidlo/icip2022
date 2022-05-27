.ONESHELL:

devserver:
	while :;
	do
		inotifywait -e modify *
		tectonic ICIP.tex
		if (( $$? ))
		then
			beep
		fi
	done
