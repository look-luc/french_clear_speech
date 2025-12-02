## This part presents a form to the user asking for some basic parameter information
form Measure Formants and Duration
	comment Sound file extension:
        	optionmenu file_type: 1
        	option .wav
        	option .aiff
	comment Max number of formants:
		integer number_of_formants 5
	comment Max formant frequency (~5500 for adult female):
		integer max_freq 5500
endform

## This part selects lets the user select a directory with the files to be measured
directory$ = chooseDirectory$ ("Choose the directory containing sound files and textgrids")
directory$ = "'directory$'" + "/" 
# The slash above will need to be changed to \ for PC users

## This part sets up an output file
resultfile$ = "'directory$'"+"formantlog.txt"
header_row$ = "vowelSAMPA" + tab$ + "Duration" + tab$ + "F0" + tab$ + "F1_Hz" + tab$ + "F2_Hz" + tab$ + "F3_Hz" + tab$ + "F0_Bark" + tab$ + "F1_Bark" + tab$ + "F2_Bark" + tab$ + "F3_Bark" + tab$ + "dist_CATcentroid_TX" + tab$ + "dist_CATcentroid_T12" + "dist_SYSTcentroid_TX" + tab$ + "dist_SYSTcentroid_T12" + tab$ + "dist_F1meanSYST_TX" + tab$ + "dist_F1meanSYST_T12" + tab$ + "dist_F2meanSYST_TX" + tab$ + "dist_F2meanSYST_T12" + tab$ + "dist_RefMinF1_LX" + tab$ + "dist_RefMinF1_LectLCS" + tab$ + "V_rms" + newline$
fileappend "'resultfile$'" 'header_row$'

# List of all the sound files in the specified directory:
Create Strings as file list... list 'directory$'*'file_type$'
number_files = Get number of strings

# This opens all the files one by one
for j from 1 to number_files
        select Strings list
        filename$ = Get string... 'j'
        Read from file... 'directory$''filename$'
        soundname$ = selected$ ("Sound")
	filedur = Get total duration
	# identify associated TextGrid
	gridfile$ = "'directory$''soundname$'.TextGrid"
	if fileReadable (gridfile$)
		Read from file... 'gridfile$'
		select TextGrid 'soundname$'
		number_intervals = Get number of intervals... 1

		# Go through each item
		for k from 1 to number_intervals
			select TextGrid 'soundname$'
			int_label$ = Get label of interval... 1 'k'
		
			#checks if interval has a labeled vowel
			if int_label$ <> ""

				# Calc start, end, and duration of interval
				intstart = Get starting point... 1 'k'
				intend = Get end point... 1 'k'
				intdur = intend - intstart
				intmid = intstart + (intdur / 2)

				# Get all the formants!
				select Sound 'soundname$'
				To Formant (burg)... 0 'number_of_formants' 'max_freq' 0.025 50
				intf1 = Get value at time... 1 'intmid' Hertz Linear
				intf2 = Get value at time... 2 'intmid' Hertz Linear
				intf3 = Get value at time... 3 'intmid' Hertz Linear

				# Dump results into a file.
				result_row$ = "'filename$'" + tab$ + "'int_label$'" + tab$ + "'intdur:3'" + tab$ + "'intf1:2'" + tab$ + "'intf2:2'" + tab$ + "'intf3:2'" + newline$
		
				fileappend "'resultfile$'" 'result_row$'
			endif
		endfor
	endif
endfor
