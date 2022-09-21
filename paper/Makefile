.ONESHELL:
.SILENT:

SOURCE := ICIP.tex
ARGS := -k

all:
	tectonic ${ARGS} ${SOURCE}

devserver:
	while :;
	do
		tectonic ${ARGS} ${SOURCE}
		# play sound on success or failure
		if (( $$? ))
		then
			# failure
			play -n -c1 synth sin 330 fade h 0.1 .2 .1 : synth sin 330 fade h 0.1 .2 0.1
		else
			# success
			play -n -c1 synth sin 440 fade h 0.1 .2 .1 : synth sin 880 fade h 0.1 .2 0.1

		fi

		# wait for file changes
		inotifywait -e modify *
	done

arxiv: all
	 zip -r icip.zip ICIP.tex ICIP.bbl spconf.sty images/

clean:
	rm icip.zip ICIP.pdf ICIP.bbl ICIP.log ICIP.blg
