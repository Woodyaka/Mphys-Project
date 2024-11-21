Quick way to launch notebook from arc:

module load anaconda

. activate Es

jupyter notebook

*New terminal*

ssh -L (port):localhost:(port) py21cb@arc4.leeds.ac.uk

CRTL + click link given in first terminal

If it asks for a token, it is given in the link you CRTL + click

Current issues:

All COSMIC figures have an issue with the colour bar. From what I know the data is the same so I am unsure of the issue.

Need to run 'SpE_Id_Algorithm.ipynb' for different criteria to completly recreate Tasha's figures, working on a batch job to run this.

Note: There is a bug in 'COSMIC_WACCM_Plotting-FINAL_CRITERIA_0.25sigma_2xMpza_1peak.ipynb' where contour_waccm is undefined and needs to be changed to contour_waccm_sum and contour_waccm_win, this is fixed in the recreation versions.
