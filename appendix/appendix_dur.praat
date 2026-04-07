form
	comment Sound file extension:
        optionmenu extension: 1
        		option .wav
        		option .aiff
	comment What tier is the appendix on:
		real tier_of_Appendix: 1
endform

directory$ = chooseDirectory$ ("Choose the directory containing sound and textgrid files")
directory$ = "'directory$'" + "/"

resultfile$ = "'directory$'"+"formantlog.txt"
header_row$ = "filename" + tab$ + "appendix type" + tab$ + "Duration"
fileappend "'resultfile$'" 'header_row$'

Create Strings as file list... list 'directory$'*'extension$'
number_files = Get number of strings

for i from 1 to number_files
		select Strings list
		filename$ = Get string... 'i'
		Read from file... 'directory$''filename$'
		soundname$ = selected$ ("Sound")
	filedur = Get total duration
	
	gridfile$ = "'directory$''soundname$'.TextGrid"
	if fileReadable (gridfile$)
		Read from file... 'gridfile$'
		interval_num = Get number of intervals... tier_of_Appendix
	
	for i from 1 to interval_num
		select TextGrid soundname$
		int_label$ = Get label of interval... tier_of_Appendix 'i'
		
		int_label_lower$ = unicodeLower$ (int_label$)
		if int_label_lower$ = "n" or int_label_lower$ = "c" or int_label_lower$ = "c+n" or int_label_lower$ = "n+c"
			start_dur = Get starting point... tier_of_Appendix 'i'
			end_dur = = Get end point... tier_of_Appendix 'i'
			duration = intend - intstart

			result_row$ = "'filename$'" + tab$ + "'int_label$'" + tab$ + "'intdur:3'" + newline$
			fileappend "'resultfile$'" 'result_row$'
			endif
		endfor
	endif
endfor