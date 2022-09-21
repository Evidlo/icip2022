.ONESHELL:
.SILENT:

SOURCE := ICIP.tex

all:
	tectonic ${SOURCE}

devserver:
	while :;
	do
		tectonic ${SOURCE}
		code=$$?

		# play sound on success or failure
		if [ "$$code" -eq "0" ]
		then
			# success
			play -n -c1 synth sin 440 fade h 0.1 .2 .1 : synth sin 880 fade h 0.1 .2 0.1
		else
			# failure
			play -n -c1 synth sin 330 fade h 0.1 .2 .1 : synth sin 330 fade h 0.1 .2 0.1

		fi

		# wait for file changes
		inotifywait -e modify *
	done

arxiv:
	 zip -r icip.zip ICIP.bbl ICIP.aux ICIP.tex IEEEbib.bst refs.bib spconf.sty images/
