Quick way to launch notebook from arc:

module load anaconda

. activate Es

jupyter notebook

*New terminal*

ssh -L (port):localhost:(port) py21cb@arc4.leeds.ac.uk

CRTL + click link given in first terminal

If it asks for a token, it is given in the link you CRTL + click

Note: There is a bug in 'COSMIC_WACCM_Plotting-FINAL_CRITERIA_0.25sigma_2xMpza_1peak.ipynb' where contour_waccm is undefined and needs to be changed to contour_waccm_sum and contour_waccm_win, this is fixed in the recreation versions.

TODO:
Note: Be careful with heights in the cli change runs.

Plot all of the climate change runs to compare against Tasha's (Normalize number of occurances)

Compare the chemical differences in the climate chnage runs against Tasha's (Anything that can effect winds/tides)

Compare the winds in the climate change runs and Tasha's
