## This part presents a form to the user asking for some basic parameter information
form Measure Formants and Duration
comment Sound file extension:
        optionmenu extension$: 1
        option .wav
        option .aiff
comment Source tier:
positive source_tier: 1
comment target tier:
positive target_tier: 2
endform


## This part selects lets the user select a directory with the files to be measured
directory$ = chooseDirectory$ ("Choose the directory containing sound files and textgrids")
directory$ = "'directory$'" + "/" 
# The slash above will need to be changed to \ for PC users


# List of all the sound files in the specified directory:
Create Strings as file list... list 'directory$'*'file_type$'
number_files = Get number of strings

# This opens all the files one by one
for j from 1 to number_files
        select Strings list
        filename$ = Get string... 'j'
        Read from file... 'directory$''filename$'
        soundname$ = selected$ ("Sound")
# identify associated TextGrid
gridfile$ = "'directory$''soundname$'.TextGrid"
if fileReadable (gridfile$)
Read from file... 'gridfile$'
select TextGrid 'soundname$'
number_intervals = Get number of intervals... source_tier

# Go through each item
for k from 1 to number_intervals
select TextGrid 'soundname$'
int_label$ = Get label of interval... source_tier 'k'
#checks if interval has a labeled vowel
if index(int_label$, "~") > 0

# Calc start, end, and duration of interval
intstart = Get starting point... source_tier 'k'
intend = Get end point... source_tier 'k'

target_interval_index = Get interval at time: target_tier_number, startTime + 0.00001
Set interval text: target_tier_number, target_interval_index, int_label$
endif
endfor
endif
save_name$ = filename$ - extension$ + "_appendix.textgrid"
    Save as text file: directory$ + save_name$
endfor