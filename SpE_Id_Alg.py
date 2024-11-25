entation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: mynewkernel
#     language: python
#     name: mynewkernel
# ---

# # FINAL ALGORITHM 

# +
#### USE THIS ONE ###

# Main notebook for Sporadic E detection algorithm for WACCM data

# -

''' INSTRUCTIONS :

- Run all notebooks from top to bottom to execute the code. 

- Things to check/change each time:

    Save_results function:
        - filename_append: Identifier for to the criteria set used to identify Es layers. 
                           '0.25sigma_2xMpza_1xpeak' is the string I used for the final criteria used for the paper
        - output_file: Define where to save the output. 

    Main Calculations:
        - run_name: Defines what model run/input data is used. Permitted to be set values (Jianfei_run, Wuhu_IonTr_run, Wuhu_IonTr_run_6m, SMin, SMax) currently. More could be added in a similar way as needed
        - ds_months_sets, Monthstr_sets, season_set : together define what seasons you're running the code for. Can be one or multiple. Common combos are left commented out for easy use
        - Criteria calculations: If changing the criteria for some reason, they need to be changed where it says "CRITERIA & Es Identifiation Calculations". 
                                 These calculations are repeated twice for calculations in local time and lon, so the criteria need to be changed in both places

Things to note:
- If a nc file exists already with the same name it will throw an error when trying to save
- Apologies for the rubbish variable names. If unsure what something is, check the attributes/dimensions in the nc variable or the code calculating the variable... Search for the variable and follow the code through
'''
import xarray as xr
import numpy as np
import cftime
import nc_time_axis
import matplotlib.pyplot as plt 
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dateutil import tz
import pytz
import time
from tqdm import tqdm
import cftime


# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Define_variables Function

# +
#Initialising arrays since they're first defined within the loop

def define_variables(lev_shape, time_shape, lat_shape, lon_shape, LTshape, time_it_shape, ds_months_shape, newlat_shape, intlat_shape, crit_freq_on):

    Mptdens_sh = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #Metal density in local time
    Mptdens_diff = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)   #Difference between Metal density and average for the timeslice

    SpEs_sh_nan = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape))  
    SpEs_sh_nan[:] = np.NaN #An array the same size as SpEs_sh filled with NaNs. Used for seletion criteria calculation

    #Mptdens_avg_temp = np.empty((lev_shape, lat_shape, 3, time_it_shape, ds_months_shape), dtype=float)  #Temporary array used in calculation rebinning from 10min -> 30 min LT bins
    Mptdens_avg_b = np.empty((lev_shape, lat_shape, LTshape, time_it_shape, ds_months_shape), dtype=float)    #10 min bins (144 lon) -> 30 min bins (48 lon)
    #-------------------------------------------------
    Mptdens_avv1_b = np.empty((lev_shape, newlat_shape, time_it_shape, ds_months_shape), dtype=float)   #Mptdens_avv1 interpolated onto 1' lat array

    Mptdens_avv1_b_5d = np.empty((lev_shape, intlat_shape, time_it_shape, ds_months_shape), dtype=float) #Average M+ density in 5' lat slices (as ft of lev)
    max_Mptdens_avv1_b_5d = np.empty((intlat_shape, time_it_shape, ds_months_shape), dtype=float)        #Max of Mptdens_avv1_b_5d over lev dim

    #Equivalent to Mptdens_avv1_b_5d and max_Mptdens_avv1_b_5d but assigned to the correct index on a normal 96-long lat axis (so can use these in selectin criteria)
    Mptdens_avv1b5d_l = np.empty((lev_shape, lat_shape, time_it_shape, ds_months_shape), dtype=float) 
    max_Mptdens_avv1b5d_l = np.empty((lat_shape, time_it_shape, ds_months_shape), dtype=float) 
    #-------------------------------------------------
    Mptdens_avg_bb = np.empty((lev_shape, newlat_shape, LTshape, time_it_shape, ds_months_shape), dtype=float)  #binned into 30 min LT bins and binned into 1' lat slices


    Mptdens_avg_bb_5d = np.empty(  (lev_shape,intlat_shape, LTshape, time_it_shape, ds_months_shape )   , dtype=float )  #Mptdens_avg_b binned into 30 min LT bins and averaged in 5' lat slices
    Mptdens_avg_bb_5d_avg = np.empty((lev_shape, intlat_shape, LTshape, ds_months_shape), dtype=float)   #Monthly avg at each height/lat/lon
    Mptdens_avg_bb_5d_avglev = np.empty((intlat_shape, LTshape, ds_months_shape), dtype=float)  #Monthly avg over all levs
    Mptdens_avg_bb_5d_dsavg = np.empty((lev_shape, intlat_shape, LTshape), dtype = float)   #Dataset avg at each height/lat/lon
    Mptdens_avg_bb_5d_dsavglev = np.empty((intlat_shape, LTshape), dtype = float) #Dataset avg over all levs


    if crit_freq_on==1:
        edens_sh = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #e density in local time
        SpEs_e = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #The e density where SpEs have been identified using Fe+ density  
        maxSpEs__e = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #max of SpEs_e in cm-3
        maxSpEs_e = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #max of SpEs_e in m-3
        foEs__m = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #Critical freq in Hz (calculated using max e- density over lev dim in m-3)
        foEs_m = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #Critical freq in MHz (calculated using max e- density over lev dim in m-3)
        foEs_m_av = np.empty((lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #Avg of foEs_m over lev dimension
        foEs_m_av_mth = np.empty((lat_shape, lon_shape, ds_months_shape), dtype = float) #Average of foEs_m_av over month
        foEs_m_av_ds = np.empty((lat_shape, lon_shape), dtype = float) #AVerage of foEs_m_av_mth over dataset (e.g. 3mth)  

    SpEs = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #SpEs occurence frequency 
    SpEs_freq = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape)) #SpE occurence in each grid space as 1s/0s
    SpEs_freq_time = np.empty((lev_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape))     #total SpE occurences in 2 week time period
    SpEs_Occ_Freq = np.empty((lev_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)  # SpEs_freq_time / no of timesteps -> occ freq as a %
    #SpEs_Occ_Freq_temp = np.empty((lev_shape, lat_shape, 3, time_it_shape, ds_months_shape), dtype=float)  #Temporary array used in the calculation for 10 min bins (144 lon) -> 30 min bins (48 lon)
    
    SpEs_freq_altsum = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape))
    SpEs_freq_altsum_time = np.empty((lat_shape, lon_shape, time_it_shape, ds_months_shape))
    SpEs_Occ_Freq_ll =  np.empty((lat_shape, lon_shape, time_it_shape, ds_months_shape))
    SpEs_Occ_Freq_llb =  np.empty((lat_shape, LTshape, time_it_shape, ds_months_shape))
    SpEs_Occ_Freq_llba =  np.empty((lat_shape, LTshape, ds_months_shape))
    SpEs_Occ_Freq_llbav = np.empty((lat_shape, LTshape))
            
    #-------------------------------------------------
    SpEs_Occ_Fr_b = np.empty((lev_shape, lat_shape, LTshape, time_it_shape, ds_months_shape), dtype=float)    #10 min bins (144 lon) -> 30 min bins (48 lon)
    SpEs_Occ_Fr_b_avg = np.empty((lev_shape, lat_shape, LTshape, ds_months_shape), dtype = float)  #Monthly avg at each height/lat/lon
    SpEs_Occ_Fr_b_dsavg = np.empty((lev_shape, lat_shape, LTshape), dtype = float)   #Dataset avg at each height/lat/lon
    SpEs_Occ_Fr_b_avgLT = np.empty((lev_shape, lat_shape, ds_months_shape), dtype = float)  #Monthly avg over all LTs
    SpEs_Occ_Fr_b_dsavgLT = np.empty((lev_shape, lat_shape), dtype = float)  #dataset avg over all LTs
    #-------------------------------------------------
    SpEs_Occ_Fr_bb = np.empty((lev_shape, newlat_shape, LTshape, time_it_shape, ds_months_shape), dtype=float) #SpEs_Occ_Fr_b interpolated onto newlat grid 1' spacing (180 long) (was 1.89')

    SpEs_Occ_Fr_bb_5d = np.empty(  (lev_shape,intlat_shape, LTshape, time_it_shape, ds_months_shape )   , dtype=float )  #SpEs_Occ_Fr_bb averaged into 5' lat slices
    SpEs_Occ_Fr_bb_5d_avg = np.empty((lev_shape, intlat_shape, LTshape, ds_months_shape), dtype=float)   #Monthly avg at each height/lat/lon
    SpEs_Occ_Fr_bb_5d_dsavg = np.empty((lev_shape, intlat_shape, LTshape), dtype = float)   #Dataset avg at each height/lat/lon
    
    SpEs_Occ_Fr_bb_5d_dsavglev = np.empty((intlat_shape, LTshape), dtype = float) #Dataset avg over all levs

    #Lat-Lon
    Mptdensns = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)  #Mptdens not shifted into local time
    Mptdens_nsavg = np.empty((lev_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)
    Mptdens_nsstd = np.empty((lev_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)
    Mptdens_nsdiff = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)   #Difference between Metal density and average
    Mptdens_nsavv1_b = np.empty((lev_shape, newlat_shape, time_it_shape, ds_months_shape), dtype=float)   #Mptdens_nsavv1 interpolated onto 1' lat array
    Mptdens_nsavv1_b_5d = np.empty((lev_shape, intlat_shape, time_it_shape, ds_months_shape), dtype=float) #Average M+ density in 5' lat slices (as ft of lev)
    max_Mptdens_nsavv1_b_5d = np.empty((intlat_shape, time_it_shape, ds_months_shape), dtype=float)        #Max of Mptdens_nsavv1_b_5d over lev dim
    #Equivalent to Mptdens_nsavv1_b_5d and max_Mptdens_nsavv1_b_5d but assigned to the correct index on a normal 96-long lat axis (so can use these in selectin criteria):
    Mptdens_nsavv1b5d_l = np.empty((lev_shape, lat_shape, time_it_shape, ds_months_shape), dtype=float) 
    max_Mptdens_nsavv1b5d_l = np.empty((lat_shape, time_it_shape, ds_months_shape), dtype=float) 

    if crit_freq_on==1:
        edensns = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)  
        SpEs_nse = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #The e density where SpEs have been identified using Fe+ density
        maxSpEs__nse = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #max of SpEs_e in cm-3
        maxSpEs_nse = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #max of SpEs_e in m-3
        foEsns__m = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #Critical freq in Hz (calculated using max e- density over lev dim in m-3)
        foEsns_m = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #Critical freq in MHz (calculated using max e- density over lev dim in m-3)
        foEsns_m_av = np.empty((lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #Avg of foEs_m over lev dimension
        foEsns_m_av_mth = np.empty((lat_shape, lon_shape, ds_months_shape), dtype = float) #Average of foEs_m_av over month
        foEsns_m_av_ds = np.empty((lat_shape, lon_shape), dtype = float) #AVerage of foEs_m_av_mth over dataset (e.g. 3mth)  

    SpEsns = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) #SpEs occurence frequency   
    SpEsns_freq_bool = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape)) #SpE occurence in each grid space as True/False 
    SpEsns_freq = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape)) #SpE occurence in each grid space as 1s/0s
    SpEsns_freq_time = np.empty((lev_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape))     #total SpE occurences in 2 week time period
    SpEsns_Occ_Freq = np.empty((lev_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)  # SpEs_freq_time / no of timesteps -> occ freq as a %
    SpEsns_Occ_Fr_avg = np.empty((lev_shape, lat_shape, lon_shape, ds_months_shape), dtype = float) 
    SpEsns_Occ_Fr_dsavg = np.empty((lev_shape, lat_shape, lon_shape), dtype = float)   

    SpEsns_freq_altsum = np.empty((time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape))
    SpEsns_freq_altsum_time = np.empty((lat_shape, lon_shape, time_it_shape, ds_months_shape))
    SpEsns_Occ_Freq_ll =  np.empty((lat_shape, lon_shape, time_it_shape, ds_months_shape))
    SpEsns_Occ_Freq_lla =  np.empty((lat_shape, lon_shape, ds_months_shape))
    SpEsns_Occ_Freq_llav = np.empty((lat_shape, lon_shape))
    
    altavg = np.empty((126, ds_months_shape), dtype = float) 
    altavg_sl = np.empty((lev_shape, ds_months_shape), dtype = float) 
    
    alt_sl_avg = np.empty((lev_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) 
    alt_sl_aavg = np.empty((lev_shape, lat_shape, lon_shape, ds_months_shape), dtype = float) 
    alt_sl_aaavg = np.empty((lev_shape, lat_shape, lon_shape), dtype = float) #(20,96,144)      #lev,lat,lon

    alt_sl_sh = np.empty((lev_shape, time_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float) 
    alt_sl_sh_avg = np.empty((lev_shape, lat_shape, lon_shape, time_it_shape, ds_months_shape), dtype = float)    
    #alt_sl_sh_avg_temp = np.empty( (lev_shape, lat_shape, 3, time_it_shape, ds_months_shape), dtype = float) 
    alt_sl_sh_avg_b = np.empty( (lev_shape, lat_shape, LTshape, time_it_shape, ds_months_shape) , dtype = float) 
    alt_sl_sh_avg_b_avg = np.empty( (lev_shape, lat_shape, LTshape, ds_months_shape) , dtype = float)
    alt_sl_sh_avg_b_dsavg = np.empty( (lev_shape, lat_shape, LTshape) , dtype = float)
    alt_sl_sh_avg_b_avgLT = np.empty( (lev_shape, lat_shape, ds_months_shape) , dtype = float)
    alt_sl_sh_avg_b_dsavgLT = np.empty( (lev_shape, lat_shape) , dtype = float)
    alt_sl_sh_avg_bb_5d = np.empty(  (lev_shape, intlat_shape, LTshape, time_it_shape, ds_months_shape )   , dtype=float ) #Mptdens_avg_b binned into 30 min LT bins and averaged in 5' lat slices
    alt_sl_sh_avg_bb_5d_avg = np.empty(  (lev_shape, intlat_shape, LTshape, ds_months_shape )   , dtype=float )
    alt_sl_sh_avg_bb = np.empty((lev_shape, newlat_shape, LTshape, time_it_shape, ds_months_shape), dtype=float)  #binned into 30 min LT bins and binned into 1' lat slices

    alt_sl_sh_avg_bb_5d_dsavg = np.empty((lev_shape, intlat_shape, LTshape), dtype = float)   #Dataset avg at each height/lat/lt
    
    
    if crit_freq_on == 1:
        return Mptdens_sh, Mptdens_diff, SpEs_sh_nan, Mptdens_avg_b, Mptdens_avv1_b, Mptdens_avv1_b_5d, max_Mptdens_avv1_b_5d, Mptdens_avv1b5d_l, max_Mptdens_avv1b5d_l, Mptdens_avg_bb, Mptdens_avg_bb_5d, Mptdens_avg_bb_5d_avg, Mptdens_avg_bb_5d_avglev, Mptdens_avg_bb_5d_dsavg, Mptdens_avg_bb_5d_dsavglev, edens_sh, SpEs_e, maxSpEs__e, maxSpEs_e, foEs__m, foEs_m, foEs_m_av, foEs_m_av_mth, foEs_m_av_ds, SpEs, SpEs_freq, SpEs_freq_time, SpEs_Occ_Freq, SpEs_Occ_Fr_b, SpEs_Occ_Fr_b_avg, SpEs_Occ_Fr_b_dsavg, SpEs_Occ_Fr_b_avgLT, SpEs_Occ_Fr_b_dsavgLT, SpEs_Occ_Fr_bb, SpEs_Occ_Fr_bb_5d, SpEs_Occ_Fr_bb_5d_avg, SpEs_Occ_Fr_bb_5d_dsavg, SpEs_Occ_Fr_bb_5d_dsavglev, Mptdensns, Mptdens_nsavg, Mptdens_nsstd, Mptdens_nsdiff, Mptdens_nsavv1_b, Mptdens_nsavv1_b_5d, max_Mptdens_nsavv1_b_5d, Mptdens_nsavv1b5d_l, max_Mptdens_nsavv1b5d_l, edensns, SpEs_nse, maxSpEs__nse, maxSpEs_nse, foEsns__m, foEsns_m, foEsns_m_av, foEsns_m_av_mth, foEsns_m_av_ds, SpEsns, SpEsns_freq_bool, SpEsns_freq, SpEsns_freq_time, SpEsns_Occ_Freq, SpEsns_Occ_Fr_avg, SpEsns_Occ_Fr_dsavg, alt_sl_avg, alt_sl_aavg, alt_sl_aaavg, alt_sl_sh, alt_sl_sh_avg, alt_sl_sh_avg_b, alt_sl_sh_avg_b_avg, alt_sl_sh_avg_b_dsavg, alt_sl_sh_avg_b_dsavgLT, alt_sl_sh_avg_b_avgLT, alt_sl_sh_avg_bb_5d, alt_sl_sh_avg_bb_5d_avg, alt_sl_sh_avg_bb_5d_dsavg, alt_sl_sh_avg_bb

    else:
        # Calculate variables when crit_freq_on is 0:
        edens_sh = None
        SpEs_e = None
        maxSpEs__e = None
        maxSpEs_e = None
        foEs__m = None
        foEs_m = None
        foEs_m_av = None
        foEs_m_av_mth = None
        foEs_m_av_ds = None
        
        edensns = None
        SpEs_nse = None
        maxSpEs__nse = None
        maxSpEs_nse = None
        foEsns__m = None
        foEsns_m = None
        foEsns_m_av = None
        foEsns_m_av_mth = None
        foEsns_m_av_ds = None

        return Mptdens_sh, Mptdens_diff, SpEs_sh_nan, Mptdens_avg_b, Mptdens_avv1_b, Mptdens_avv1_b_5d, max_Mptdens_avv1_b_5d, Mptdens_avv1b5d_l, max_Mptdens_avv1b5d_l, Mptdens_avg_bb, Mptdens_avg_bb_5d, Mptdens_avg_bb_5d_avg, Mptdens_avg_bb_5d_avglev, Mptdens_avg_bb_5d_dsavg, Mptdens_avg_bb_5d_dsavglev, edens_sh, SpEs_e, maxSpEs__e, maxSpEs_e, foEs__m, foEs_m, foEs_m_av, foEs_m_av_mth, foEs_m_av_ds, SpEs, SpEs_freq, SpEs_freq_time, SpEs_Occ_Freq, SpEs_Occ_Fr_b, SpEs_Occ_Fr_b_avg, SpEs_Occ_Fr_b_dsavg, SpEs_Occ_Fr_b_avgLT, SpEs_Occ_Fr_b_dsavgLT, SpEs_Occ_Fr_bb, SpEs_Occ_Fr_bb_5d, SpEs_Occ_Fr_bb_5d_avg, SpEs_Occ_Fr_bb_5d_dsavg, SpEs_Occ_Fr_bb_5d_dsavglev, Mptdensns, Mptdens_nsavg, Mptdens_nsstd, Mptdens_nsdiff, Mptdens_nsavv1_b, Mptdens_nsavv1_b_5d, max_Mptdens_nsavv1_b_5d, Mptdens_nsavv1b5d_l, max_Mptdens_nsavv1b5d_l, edensns, SpEs_nse, maxSpEs__nse, maxSpEs_nse, foEsns__m, foEsns_m, foEsns_m_av, foEsns_m_av_mth, foEsns_m_av_ds, SpEsns, SpEsns_freq_bool, SpEsns_freq, SpEsns_freq_time, SpEsns_Occ_Freq, SpEsns_Occ_Fr_avg, SpEsns_Occ_Fr_dsavg, altavg, altavg_sl, alt_sl_avg, alt_sl_aavg, alt_sl_aaavg, alt_sl_sh, alt_sl_sh_avg, alt_sl_sh_avg_b, alt_sl_sh_avg_b_avg, alt_sl_sh_avg_b_dsavg, alt_sl_sh_avg_b_dsavgLT, alt_sl_sh_avg_b_avgLT, alt_sl_sh_avg_bb_5d, alt_sl_sh_avg_bb_5d_avg, alt_sl_sh_avg_bb_5d_dsavg, alt_sl_sh_avg_bb, SpEs_freq_altsum, SpEs_freq_altsum_time, SpEs_Occ_Freq_ll, SpEs_Occ_Freq_llb, SpEs_Occ_Freq_llba, SpEs_Occ_Freq_llbav, SpEsns_freq_altsum,SpEsns_freq_altsum_time,SpEsns_Occ_Freq_ll,SpEsns_Occ_Freq_lla,SpEsns_Occ_Freq_llav
        

# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Save_results Function

# +
# Save output from SpE algorithm to nc files for each season

def save_results(run_name, Monthfolderstr, lev, lev_sl, timear, lat, intlat, lon, LTar, LTlong, time_ar_2wk, ds_months, Zavg_sl, altavg, altavg_sl, times_str_min, times_str_max, SpEs_Occ_Fr_b_dsavg, SpEs_Occ_Fr_b_avg, SpEs_Occ_Fr_bb_5d_dsavg, SpEsns_Occ_Fr_dsavg, SpEs, alt_sl_sh_avg_b_dsavg, alt_sl_sh_avg_b_dsavgLT, alt_sl_sh_avg_bb_5d_dsavg, Mptdens_avv1_b_5d, Mptdens_std, Mptdens_nsstd, SpEsns_freq_time, SpEs_freq_time, Mptdens_avg, Mptdens_nsdiff, Mptdens_nsavg, SpEs_Occ_Freq_llbav, SpEsns_Occ_Freq_llav, SpEsns_Occ_Freq_lat ):
    
    filename_append = '0.25sigma_2xMpza_1xpeak' 
    
    output_file = f'Nc_Files/SpE_Output/SMin/{str(run_name)}_SpE_Output_{str(Monthfolderstr)}_90-150km_{filename_append}.nc'


    
    with nc.Dataset(output_file, 'w') as dataset:

        # Create dimensions
        dataset.createDimension('lev', len(lev))
        dataset.createDimension('lev_sl', len(lev_sl))
        dataset.createDimension('time', len(timear))
        dataset.createDimension('lat', len(lat))
        dataset.createDimension('latsl', len(intlat))
        dataset.createDimension('lon', len(lon))
        dataset.createDimension('LT', len(LTar))
        dataset.createDimension('LT_L', len(LTlong))
        dataset.createDimension('timesl', 2)
        dataset.createDimension('mth', 3)

        # Create coordinate variables
        lev_coord = dataset.createVariable('lev', 'f8', ('lev',))
        lev_sl_coord = dataset.createVariable('lev_sl', 'f8', ('lev_sl',))
        time_coord = dataset.createVariable('time', 'f8', ('time',))
        lat_coord = dataset.createVariable('lat', 'f8', ('lat',))
        latsl_coord = dataset.createVariable('latsl', 'f8', ('latsl',))
        lon_coord = dataset.createVariable('lon', 'f8', ('lon',))
        LT_coord = dataset.createVariable('LT', 'f8', ('LT',))
        LT_L_coord = dataset.createVariable('LT_L', 'f8', ('LT_L',))
        timesl_coord = dataset.createVariable('timesl', 'f8', ('timesl',))
        mth_coord = dataset.createVariable('mth', 'f8', ('mth',))

        # Assign values to coordinate variables
        lev_coord[:] = lev[:]
        lev_sl_coord[:] = lev_sl[:]
        time_coord[:] = timear[:]
        lat_coord[:] = lat[:]
        latsl_coord[:] = intlat[:]
        lon_coord[:] = lon[:]
        LT_coord[:] = LTar[:]  
        LT_L_coord[:] = LTlong[:]   
        timesl_coord[:] = time_ar_2wk[:]
        mth_coord[:] = ds_months[:]

        
        # Add attributes to coordinate variables
        lev_coord.long_name = 'vertical level'
        lev_coord.units = 'hPa'
        
        lev_sl_coord.long_name = 'vertical level sliced lev[42:60] (~90-130km geometric alt)'
        lev_sl_coord.units = 'hPa'
        
        time_coord.long_name = 'time'
        time_coord.units = 'days since start of 2 week time slice (0-335)'
        
        lat_coord.long_name = 'latitude'
        lat_coord.units = 'degrees_north'
        
        latsl_coord.long_name = 'latitude grid sliced in 5deg slices (array 0-35)'
        latsl_coord.units = 'degrees_north'
        
        lon_coord.long_name = 'longitude'
        lon_coord.units = 'degrees_east'
        
        LT_coord.long_name = 'Local Time, 48 x 30 min bins'
        LT_coord.units = 'hours'
        
        LT_L_coord.long_name = 'Local Time, 144 bins (WACCM resolution)'
        LT_L_coord.units = 'hours'
        
        timesl_coord.long_name = 'time slice identifier, 2 timeslices per month'
        timesl_coord.units = 'arrary [0,1]'
        
        mth_coord.long_name = 'month identifier for 3 month seasonal period'
        #mth_coord.units = ''
        
        
        
        # Create and save arrays as variables

        Zavg_sl_v = dataset.createVariable('Zavg_sl', 'f8', ('lev_sl',))
        Zavg_sl_v[:] = Zavg_sl[:]
        Zavg_sl_v.setncattr('units', 'km')
        Zavg_sl_v.setncattr('description', 'Global average Geopotential height, sliced between 90-130km')

        altavg_v = dataset.createVariable('altavg', 'f8', ('lev',))
        altavg_v[:] = altavg[:]
        altavg_v.setncattr('units', 'km')
        altavg_v.setncattr('description', 'Global average geometric altitude')
        
        altavg_sl_v = dataset.createVariable('altavg_sl', 'f8', ('lev_sl',))
        altavg_sl_v[:] = altavg_sl[:]
        altavg_sl_v.setncattr('units', 'km')
        altavg_sl_v.setncattr('description', 'Global average geometric altitude, sliced between 90-130km')

        times_str_min_v = dataset.createVariable('times_str_min', 'S21', ('mth', 'timesl'))
        times_str_min_v[:] = times_str_min[:]
        times_str_min_v.setncattr('description', 'Timeslice start datetime')

        times_str_max_v = dataset.createVariable('times_str_max', 'S21', ('mth', 'timesl'))
        times_str_max_v[:] = times_str_max[:]
        times_str_max_v.setncattr('description', 'Timeslice end datetime')

        SpEsns_Occ_Freq_lat_v = dataset.createVariable('SpEsns_Occ_Freq_lat', 'f8', ('lat',))
        SpEsns_Occ_Freq_lat_v[:] = SpEsns_Occ_Freq_lat[:]
        SpEsns_Occ_Freq_lat_v.setncattr('units', '%')
        SpEsns_Occ_Freq_lat_v.setncattr('description', 'Occ Freq - ds average over all heights')
        
        SpEs_Occ_Fr_b_dsavg_v = dataset.createVariable('SpEs_Occ_Fr_b_dsavg', 'f8', ('lev_sl', 'lat', 'LT'))
        SpEs_Occ_Fr_b_dsavg_v[:] = SpEs_Occ_Fr_b_dsavg[:]
        SpEs_Occ_Fr_b_dsavg_v.setncattr('units', '%')
        SpEs_Occ_Fr_b_dsavg_v.setncattr('description', 'Occ Freq - ds average at each height')

        alt_sl_sh_avg_b_dsavg_v = dataset.createVariable('alt_sl_sh_avg_b_dsavg', 'f8', ('lev_sl', 'lat', 'LT'))
        alt_sl_sh_avg_b_dsavg_v[:] = alt_sl_sh_avg_b_dsavg[:]
        alt_sl_sh_avg_b_dsavg_v.setncattr('units', 'km')
        alt_sl_sh_avg_b_dsavg_v.setncattr('description', 'alt - ds average at each height')
        
        SpEs_Occ_Fr_b_avg_v = dataset.createVariable('SpEs_Occ_Fr_b_avg', 'f8', ('lev_sl', 'lat', 'LT', 'mth'))
        SpEs_Occ_Fr_b_avg_v[:] = SpEs_Occ_Fr_b_avg[:]
        SpEs_Occ_Fr_b_avg_v.setncattr('units', '%')
        SpEs_Occ_Fr_b_avg_v.setncattr('description', 'Occ Freq - monthly average at each height')

        SpEs_Occ_Fr_b_dsavgLT_v = dataset.createVariable('SpEs_Occ_Fr_b_dsavgLT', 'f8', ('lev_sl', 'lat'))
        SpEs_Occ_Fr_b_dsavgLT_v[:] = SpEs_Occ_Fr_b_dsavgLT[:]
        SpEs_Occ_Fr_b_dsavgLT_v.setncattr('units', '%')
        SpEs_Occ_Fr_b_dsavgLT_v.setncattr('description', 'Occ Freq - ds average over all LTs')
        
        alt_sl_sh_avg_b_dsavgLT_v = dataset.createVariable('alt_sl_sh_avg_b_dsavgLT', 'f8', ('lev_sl', 'lat'))
        alt_sl_sh_avg_b_dsavgLT_v[:] = alt_sl_sh_avg_b_dsavgLT[:]
        alt_sl_sh_avg_b_dsavgLT_v.setncattr('units', 'km')
        alt_sl_sh_avg_b_dsavgLT_v.setncattr('description', 'alt - ds average over all LTs')

        SpEs_Occ_Fr_b_avgLT_v = dataset.createVariable('SpEs_Occ_Fr_b_avgLT', 'f8', ('lev_sl', 'lat', 'mth'))
        SpEs_Occ_Fr_b_avgLT_v[:] = SpEs_Occ_Fr_b_avgLT[:]
        SpEs_Occ_Fr_b_avgLT_v.setncattr('units', '%')
        SpEs_Occ_Fr_b_avgLT_v.setncattr('description', 'Occ Freq - monthly average over all LTs')

        SpEs_Occ_Fr_bb_5d_dsavg_v = dataset.createVariable('SpEs_Occ_Fr_bb_5d_dsavg', 'f8', ('lev_sl', 'latsl', 'LT'))
        SpEs_Occ_Fr_bb_5d_dsavg_v[:] = SpEs_Occ_Fr_bb_5d_dsavg[:]
        SpEs_Occ_Fr_bb_5d_dsavg_v.setncattr('units', '%')
        SpEs_Occ_Fr_bb_5d_dsavg_v.setncattr('description', 'Occ Freq - ds average at each height in 5deg lat band')

        alt_sl_sh_avg_bb_5d_dsavg_v = dataset.createVariable('alt_sl_sh_avg_bb_5d_dsavg', 'f8', ('lev_sl', 'latsl', 'LT'))
        alt_sl_sh_avg_bb_5d_dsavg_v[:] = alt_sl_sh_avg_bb_5d_dsavg[:]
        alt_sl_sh_avg_bb_5d_dsavg_v.setncattr('units', 'km')
        alt_sl_sh_avg_bb_5d_dsavg_v.setncattr('description', 'Alt - ds average lev-LT in 5deg lat band')

        SpEsns_Occ_Fr_dsavg_v = dataset.createVariable('SpEsns_Occ_Fr_dsavg', 'f8', ('lev_sl', 'lat', 'lon'))
        SpEsns_Occ_Fr_dsavg_v[:] = SpEsns_Occ_Fr_dsavg[:]
        SpEsns_Occ_Fr_dsavg_v.setncattr('units', '%')
        SpEsns_Occ_Fr_dsavg_v.setncattr('description', 'Occ Freq - ds average at each height')
 
        SpEs_Occ_Freq_llbav_v = dataset.createVariable('SpEs_Occ_Freq_llbav', 'f8', ( 'lat', 'LT'))
        SpEs_Occ_Freq_llbav_v[:] = SpEs_Occ_Freq_llbav[:]
        SpEs_Occ_Freq_llbav_v.setncattr('units', '%')
        SpEs_Occ_Freq_llbav_v.setncattr('description', 'Occ Freq - ds average (lat,LT), alternative way of counting stats over height')

        SpEsns_Occ_Freq_llav_v = dataset.createVariable('SpEsns_Occ_Freq_llav', 'f8', ( 'lat', 'lon'))
        SpEsns_Occ_Freq_llav_v[:] = SpEsns_Occ_Freq_llav[:]
        SpEsns_Occ_Freq_llav_v.setncattr('units', '%')
        SpEsns_Occ_Freq_llav_v.setncattr('description', 'Occ Freq - ds average (lat,lon), alternative way of counting stats over height')
        
        
        
        # LOCAL TIME
        SpEs_v = dataset.createVariable('SpEs', 'f8', ('lev_sl', 'time', 'lat', 'LT_L', 'timesl', 'mth'))
        SpEs_v[:] = SpEs[:]
        SpEs_v.setncattr('units', '%')
        SpEs_v.setncattr('description', 'Metal density (144xLT) where Es layer identified (NaNs elsewhere)')

        Mptdens_sh_v = dataset.createVariable('Mptdens_sh', 'f8', ('lev_sl', 'time', 'lat', 'LT_L', 'timesl', 'mth'))
        Mptdens_sh_v[:] = Mptdens_sh[:]
        Mptdens_sh_v.setncattr('units', 'cm-3')
        Mptdens_sh_v.setncattr('description', 'Metal density (144xLT)')

        
        # LAT-LON
        SpEsns_v = dataset.createVariable('SpEsns', 'f8', ('lev_sl', 'time', 'lat', 'lon', 'timesl', 'mth'))
        SpEsns_v[:] = SpEsns[:]
        SpEsns_v.setncattr('units', '%')
        SpEsns_v.setncattr('description', 'Metal density (lon) where Es layer identified (NaNs elsewhere)')

        Mptdensns_v = dataset.createVariable('Mptdensns', 'f8', ('lev_sl', 'time', 'lat', 'lon', 'timesl', 'mth'))
        Mptdensns_v[:] = Mptdensns[:]
        Mptdensns_v.setncattr('units', 'cm-3')
        Mptdensns_v.setncattr('description', 'Metal density (lon)')
        
        Mptdens_nsstd_v = dataset.createVariable('Mptdens_nsstd', 'f8', ('lev_sl', 'lat', 'lon', 'timesl', 'mth'))
        Mptdens_nsstd_v[:] = Mptdens_nsstd[:]
        Mptdens_nsstd_v.setncattr('units', 'cm-3')
        Mptdens_nsstd_v.setncattr('description', 'standard deviation of the M+ layer in each 2wk time slice')
            
        Mptdens_nsdiff_v = dataset.createVariable('Mptdens_nsdiff', 'f8', ('lev_sl', 'time', 'lat', 'lon', 'timesl', 'mth'))       
        Mptdens_nsdiff_v[:] = Mptdens_nsdiff[:]
        Mptdens_nsdiff_v.setncattr('units', 'cm-3')
        Mptdens_nsdiff_v.setncattr('description', 'Difference between total M+ density and the average M+ layer in each 2wk time slice')
          
        
        Mptdens_nsavg_v = dataset.createVariable('Mptdens_nsavg', 'f8', ('lev_sl', 'lat', 'lon', 'timesl', 'mth'))
        Mptdens_nsavg_v[:] = Mptdens_nsavg[:]
        Mptdens_nsavg_v.setncattr('units', 'cm-3')
        Mptdens_nsavg_v.setncattr('description', 'avg M+ layer for each 2wk time slice')
        
    
        Mptdens_avv1_b_5d_v = dataset.createVariable('Mptdens_avv1_b_5d', 'f8', ('lev_sl', 'latsl', 'timesl', 'mth'))
        Mptdens_avv1_b_5d_v[:] = Mptdens_avv1_b_5d[:]
        Mptdens_avv1_b_5d_v.setncattr('units', 'cm-3')
        Mptdens_avv1_b_5d_v.setncattr('description', 'zonal avg M layer in 5deg lat slices (for each 2wk time slice)')
        
        Mptdens_std_v = dataset.createVariable('Mptdens_std', 'f8', ('lev_sl', 'lat', 'LT_L', 'timesl', 'mth'))
        Mptdens_std_v[:] = Mptdens_std[:]
        Mptdens_std_v.setncattr('units', 'cm-3')
        Mptdens_std_v.setncattr('description', 'standard deviation of the M+ layer in each 2wk time slice')
        
        Mptdens_avg_v = dataset.createVariable('Mptdens_avg', 'f8', ('lev_sl', 'lat', 'LT_L', 'timesl', 'mth'))
        Mptdens_avg_v[:] = Mptdens_avg[:]
        Mptdens_avg_v.setncattr('units', 'cm-3')
        Mptdens_avg_v.setncattr('description', 'avg M+ layer for each 2wk time slice')
        
        
        #No of occurences in each grid box in 2wk period (in lon and LT)
        SpEs_freq_time_v = dataset.createVariable('SpEs_freq_time', 'f8', ('lev_sl', 'lat', 'LT_L', 'timesl', 'mth'))
        SpEs_freq_time_v[:] = SpEs_freq_time[:]
        SpEs_freq_time_v.setncattr('units', 'cm-3')
        SpEs_freq_time_v.setncattr('description', 'total SpE occurences in 2 week time period (on LT axis (144 long)')
        
        SpEsns_freq_time_v = dataset.createVariable('SpEsns_freq_time', 'f8', ('lev_sl', 'lat', 'lon', 'timesl', 'mth'))
        SpEsns_freq_time_v[:] = SpEsns_freq_time[:]
        SpEsns_freq_time_v.setncattr('units', 'cm-3')
        SpEsns_freq_time_v.setncattr('description', 'total SpE occurences in 2 week time period (on lon axis)')
        
        
        
    print(f"Results saved to {output_file}")
    

# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Dimension sizes & other parameters

# +
#Define various parameters related to the dimension sizes or diff arrays used in the algorithm

def setup_parameters():
    # Set time parameters - No of time samples to iterate over (2x 2 week periods)
    time_it_shape = 2   
    time_ind_2wk_min = [0, 336]
    time_ind_2wk_max = [335, 671]

    # Set time parameters - No of timesteps in one time sample (2 week period)
    time_shape = 336
    timear = np.arange(0, time_shape)  # 2 week period (hourly data)

    # Lon parameters
    lon_shape = 144
    lonar = np.arange(0, lon_shape)     
    
    LTshape = 48 #72
    it_arr = np.arange(0, LTshape)  # array 0->LTshape   

    # Bin midpoints covering whole range
    # LTar = np.arange( ((24/LTshape)/2) , 24 , (24/LTshape))  
    # LTlong = np.arange( ((24/lon_shape)/2) , 24 , (24/lon_shape)) #0.0833 to 23.9166, x144 inds 10 min bins
    
    # Using left edge
    LTar = np.arange(0, 24, 24/48)  #0-23.5, 48 long
    LTlong = np.arange(0, 24, 24/144)   #0-23.833, 144 long
    
    
    # Lat parameters
    lat_shape = 96
    latar = np.arange(0, lat_shape)

    # Lat grid in 1' increments
    newlat = np.arange(-89.5, 90.5, 1)
    newlat_shape = 180

    # Lat grid in 5' slices (each index is centrepoint of slice)
    intlat = np.arange(-87.5, 92.5, 5)
    intlat_shape = 36
    intlat_ar = np.arange(0, intlat_shape)  # array 0->35

    
    # Average altitudes over the year (km) - geopotential height and geometric altitude, at indices listed:
        # geopotential height:
        # zavglist41 = 133.23580932617188
        # zavglist42 = 128.60214233398438
        # zavglist60 = 89.4076919555664
        #geometric altitude:
        # altavglist41 = 136.07879638671875
        # altavglist42 = 131.2488250732422 ***
        # altavglist60 = 90.67900848388672 ***
    # 90-130km --> To slice lev dim using indices 42 and 60 inclusive and plot on geometric altitude 
    #90-150km --> #to slice between indices 38 and 60
    
    # Slice arrays (lev, altitude) between chosen range from above
    #90-150km
    lev_sl_idx_min = 38 #42      
    lev_sl_idx_max = 60      
    lev_shape = (lev_sl_idx_max - lev_sl_idx_min) + 1 
    levar = np.arange(0, lev_shape)          

    # Create an array with offset needed for each UT time step (24h period)
    # Offset by 15 degrees lon each time, lon axis is in 2.5 degree intervals
    offset = np.arange(0, 24) * 15 / 2.5 
    offset = offset.astype(int)
    offset = np.tile(offset, 14)  # tile the array for 2 weeks of 1hrly timesteps 2wks * 168timesteps = 336

    # Return all the defined arrays as a tuple
    return time_it_shape, time_ind_2wk_min, time_ind_2wk_max, time_shape, timear, lon_shape, LTshape, LTlong, lonar, LTar, LTlong, it_arr, lat_shape, latar, newlat, newlat_shape, intlat, intlat_shape, intlat_ar, lev_sl_idx_min, lev_sl_idx_max, lev_shape, levar, offset

# -

# # Main calculations - for SpEs and crit freq

# +
start_time = time.process_time() 


#////////////////////////////  Things to check/change before running  ////////////////////////////////////

# Critical frequency on/off switch. Turn on (set =1) if wanting to calculate the critical frequency and other related parameters. 
# Check before using as haven't used in a while and may need updating
crit_freq_on = 0   


run_name = 'SMin'             #Set to the relevant run name to change input files. Currently working for the runs below:
                                    #- Wuhu_IonTr_run - Wuhu's 3 metal run (Fe, Mg and Na). Equivalent to Jianfei_run but done at Leeds. Main dataset for analysis for SpE paper
                                    #- Jianfei_run - equivalent to Wuhu_IonTr_run from Wu et al 2021
                                    #- Wuhu_IonTr_run_6m - Wuhu's 6 metal run (Fe, Mg, Na, Si, K, Ca) - gives similar results to 3 metal run
                                    #- SMin and SMax - solar runs

            
# Set season/months to use for analysis. Can do one/multiple/all seasons (3 month groups):

# ds_months_sets = [['12','01','02'], ['03','04','05'], ['06','07','08'], ['09','10','11']]
# Monthstr_sets = [['Dec', 'Jan', 'Feb'], ['Mar', 'Apr', 'May'], ['Jun', 'Jul', 'Aug'], ['Sep', 'Oct', 'Nov']]
# season_set = ['winter', 'spring', 'summer', 'autumn']

# ds_months_sets = [['12','01','02'], ['06','07','08'], ['09','10','11']]
# Monthstr_sets = [['Dec', 'Jan', 'Feb'],  ['Jun', 'Jul', 'Aug'], ['Sep', 'Oct', 'Nov']]
# season_set = ['winter', 'summer', 'autumn']

# ds_months_sets = [['12','01','02'], ['03','04','05'], ['09','10','11']]
# Monthstr_sets = [['Dec', 'Jan', 'Feb'], ['Mar', 'Apr', 'May'], ['Sep', 'Oct', 'Nov']]
# season_set = ['winter', 'spring', 'autumn']


# ds_months_sets = [ ['03','04','05'], ['12','01','02'] ]
# Monthstr_sets = [['Mar', 'Apr', 'May'], ['Dec', 'Jan', 'Feb']  ]
# season_set = [ 'spring', 'winter' ]

# ds_months_sets = [ ['06','07','08'], ['09','10','11'] ]
# Monthstr_sets = [ ['Jun', 'Jul', 'Aug'], ['Sep', 'Oct', 'Nov'] ]
# season_set = [ 'summer', 'autumn' ]

# ds_months_sets = [ ['03','04','05'], ['06','07','08'] ]
# Monthstr_sets = [['Mar', 'Apr', 'May'], ['Jun', 'Jul', 'Aug']  ]
# season_set = [ 'spring', 'summer' ]

# ds_months_sets = [ ['09','10','11'], ['12','01','02'] ]
# Monthstr_sets = [ ['Sep', 'Oct', 'Nov'], ['Dec', 'Jan', 'Feb'] ]
# season_set = [ 'autumn', 'winter'  ]

# ds_months_sets = [ ['03','04','05'] ]
# Monthstr_sets = [ ['Mar', 'Apr', 'May'] ]
# season_set = [ 'spring']

# ds_months_sets = [['06','07','08']]
# Monthstr_sets = [['Jun','Jul','Aug']]
# season_set = ['summer']

ds_months_sets = [['09','10','11']]
Monthstr_sets = [['Sep', 'Oct', 'Nov']]
season_set = ['autumn']

# ds_months_sets = [['12','01','02']]
# Monthstr_sets = [['Dec', 'Jan', 'Feb']]
# season_set = ['winter']

#////////////////////////////////////////////////  end  /////////////////////////////////////////////////////////

        



    
#/////////////////////////////////////////////////////////////////////////////////////////////////////////

for set_idx in range(len(ds_months_sets)):  #loop through seasons
    ds_months = ds_months_sets[set_idx]
    ds_months_shape = len(ds_months)
    Monthstr = Monthstr_sets[set_idx]
    print(season_set[set_idx])

    #--------------------------------------------------------------------------------------------------
    time_ar_2wk = np.arange(0,2)
    
    #Initialising time string arrays used for figure labelling 
    if ds_months_shape == 1:
        times_str_min = np.array( (time_ar_2wk) , dtype=str  ) 
        times_str_max = np.array( (time_ar_2wk) , dtype=str  ) 
    elif ds_months_shape ==2:
        times_str_min = np.array( (time_ar_2wk,time_ar_2wk) , dtype=str  ) 
        times_str_max = np.array( (time_ar_2wk,time_ar_2wk) , dtype=str  ) 
    elif ds_months_shape ==3:   
        times_str_min = np.array( (time_ar_2wk,time_ar_2wk,time_ar_2wk) , dtype=str  ) 
        times_str_max = np.array( (time_ar_2wk,time_ar_2wk,time_ar_2wk) , dtype=str  ) 
    #--------------------------------------------------------------------------------------------------
    #Creating string of relevant months for output file path
    Monthfolderstr = Monthstr[0] + '-' + Monthstr[-1]
    if ds_months_shape == 1:
        Monthfolderstr = Monthstr[0]

    #Creating array for iterating through x3 months in each season
    ds_months_ar = np.arange(0,ds_months_shape)   #[0,1,2]
    #--------------------------------------------------------------------------------------------------

    #Call setup_parameters() function
    time_it_shape, time_ind_2wk_min, time_ind_2wk_max, time_shape, timear, lon_shape, LTshape, LTlong, lonar, LTar, LTlong, it_arr, lat_shape, latar, newlat, newlat_shape, intlat, intlat_shape, intlat_ar, lev_sl_idx_min, lev_sl_idx_max, lev_shape, levar, offset = setup_parameters()
    
    #------------------ Variables for selection criteria ----------------------------------------------
    sigma_val = 1   #1sigma~68%   (1.5sigma~87%   2sigma~95%)
    sigma_val_str = str(sigma_val)

    #call define_variables() function
    Mptdens_sh, Mptdens_diff, SpEs_sh_nan, Mptdens_avg_b, Mptdens_avv1_b, Mptdens_avv1_b_5d, max_Mptdens_avv1_b_5d, Mptdens_avv1b5d_l, max_Mptdens_avv1b5d_l, Mptdens_avg_bb, Mptdens_avg_bb_5d, Mptdens_avg_bb_5d_avg, Mptdens_avg_bb_5d_avglev, Mptdens_avg_bb_5d_dsavg, Mptdens_avg_bb_5d_dsavglev, edens_sh, SpEs_e, maxSpEs__e, maxSpEs_e, foEs__m, foEs_m, foEs_m_av, foEs_m_av_mth, foEs_m_av_ds, SpEs, SpEs_freq, SpEs_freq_time, SpEs_Occ_Freq, SpEs_Occ_Fr_b, SpEs_Occ_Fr_b_avg, SpEs_Occ_Fr_b_dsavg, SpEs_Occ_Fr_b_avgLT, SpEs_Occ_Fr_b_dsavgLT, SpEs_Occ_Fr_bb, SpEs_Occ_Fr_bb_5d, SpEs_Occ_Fr_bb_5d_avg, SpEs_Occ_Fr_bb_5d_dsavg, SpEs_Occ_Fr_bb_5d_dsavglev, Mptdensns, Mptdens_nsavg, Mptdens_nsstd, Mptdens_nsdiff, Mptdens_nsavv1_b, Mptdens_nsavv1_b_5d, max_Mptdens_nsavv1_b_5d, Mptdens_nsavv1b5d_l, max_Mptdens_nsavv1b5d_l, edensns, SpEs_nse, maxSpEs__nse, maxSpEs_nse, foEsns__m, foEsns_m, foEsns_m_av, foEsns_m_av_mth, foEsns_m_av_ds, SpEsns, SpEsns_freq_bool, SpEsns_freq, SpEsns_freq_time, SpEsns_Occ_Freq, SpEsns_Occ_Fr_avg, SpEsns_Occ_Fr_dsavg, altavg, altavg_sl, alt_sl_avg, alt_sl_aavg, alt_sl_aaavg, alt_sl_sh, alt_sl_sh_avg, alt_sl_sh_avg_b, alt_sl_sh_avg_b_avg, alt_sl_sh_avg_b_dsavg, alt_sl_sh_avg_b_dsavgLT, alt_sl_sh_avg_b_avgLT, alt_sl_sh_avg_bb_5d, alt_sl_sh_avg_bb_5d_avg, alt_sl_sh_avg_bb_5d_dsavg, alt_sl_sh_avg_bb ,SpEs_freq_altsum, SpEs_freq_altsum_time, SpEs_Occ_Freq_ll, SpEs_Occ_Freq_llb, SpEs_Occ_Freq_llba, SpEs_Occ_Freq_llbav, SpEsns_freq_altsum, SpEsns_freq_altsum_time, SpEsns_Occ_Freq_ll, SpEsns_Occ_Freq_lla, SpEsns_Occ_Freq_llav  = define_variables(lev_shape, time_shape, lat_shape, lon_shape, LTshape, time_it_shape, ds_months_shape, newlat_shape, intlat_shape, crit_freq_on)
 
    #--------------------------------------------------------------------------------------------------

    for ids in ds_months_ar:  #loop through months in season
        #===================================================================================================
        loop_start_time = time.process_time()   #calculates timing for terminal output
        #===================================================================================================
        #Define file paths for data source and open dataset
        if run_name=='Jianfei_run':
            file1name =f'Nc_Files/Jianfei_WACCMX_files/waccmx_Fe_Fep_{ds_months[ids]}.nc'
            ds = xr.open_dataset(file1name)
        elif run_name=='Wuhu_IonTr_run':
            file1name=f'Nc_Files/ACP_CESM213_FX2000_f19_f19_mg16_Na_Fe_Mg_iontransport/ACP_CESM213_FX2000_f19_f19_mg16_Na_Fe_Mg_iontransport.cam.h2.0001-{ds_months[ids]}-*.nc'
            ds = xr.open_mfdataset(file1name)
        elif run_name=='Wuhu_IonTr_run_6m':
            file1name=f'Nc_Files/CESM213_FX2000_f19_f19_mg16_Na_Fe_Mg_Si_Ca_K_iontransport/CESM213_FX2000_f19_f19_mg16_Na_Fe_Mg_Si_Ca_K_iontransport.cam.h2.0001-{ds_months[ids]}-*.nc' 
            ds = xr.open_mfdataset(file1name)
        elif run_name=='SMax':
            file1name=f'Nc_Files/SMax_3M_FX2000_f19_f19mg16/SMax_3M_FX2000_f19f19mg16.cam.h1.0001-{ds_months[ids]}-*.nc'
            ds = xr.open_mfdataset(file1name)
        elif run_name=='SMin':
            file1name=f'Nc_Files/SMin_3M_FX2000_f19_f19mg16/SMin_3M_FX2000_f19f19mg16.cam.h1.0001-{ds_months[ids]}-*.nc'
            ds = xr.open_mfdataset(file1name)
        #===================================================================================================
        #Define dimension variables
        print(f'Dataset: Month {str(ds_months[ids])}') 

        timee = ds.variables['time']
        start_cftime_date = f'0001-{str(ds_months[ids])}-01'
        times = xr.cftime_range(start=start_cftime_date, periods=672, freq="1H", calendar="noleap")    

        lon = ds.variables['lon']
        lat = ds.variables['lat']
        lev = ds.variables['lev']
        dst = ds.transpose("lev", ...)
        #===================================================================================================
        #Define more variables that may or may not be from a different file depending ont the run
        if run_name=='Jianfei_run':
            file2name='Nc_Files/Jianfei_WACCMX_files/waccmx_Z3_T_e_' + ds_months[ids] + '.nc' 
            ds2 = xr.open_dataset(file2name) 
            ds2t = ds2.transpose("lev", ...)
            temp = ds2t.variables['T']
            geopH = ds2t.variables['Z3'] / 1000 #m-> km
            if crit_freq_on==1:
                elect = ds2t.variables['e']
        else:
            temp = dst.variables['T']
            elect = dst.variables['e']
            geopH = dst.variables['Z3'] / 1000 #m-> km    #(126, 744, 96, 144)

        Fept = dst.variables['Fep']
        Mgpt = dst.variables['Mgp']
        Napt = dst.variables['Nap']
        
        Zavg = geopH.mean(('time','lat', 'lon'))   #Global average of geopotential height (126)

        #Convert geopotential height to geometric altitude
        Re = 6378 #km
        alt = (geopH*Re)/(Re-geopH)
        altavg[:,ids] = alt.mean(('time','lat', 'lon')) #global average

        #===================================================================================================

        # Slice arrays (lev & alt) between chosen geometric altitude range & print
        lev_sl = lev[lev_sl_idx_min:lev_sl_idx_max+1]
        Zavg_sl = Zavg[lev_sl_idx_min:lev_sl_idx_max+1] 
        altavg_sl[:,ids] = altavg[lev_sl_idx_min:lev_sl_idx_max+1,ids]
        
        alt_sl = alt[lev_sl_idx_min:lev_sl_idx_max+1,:,:,:]    

        print(f'    Array lev = {str("%.1e" % lev[lev_sl_idx_max])}hPa : {str("%.1e" % lev[lev_sl_idx_min])}hPa'
                   + f' (approx {str("%.0f" % Zavg[lev_sl_idx_max])}km : {str("%.0f" % Zavg[lev_sl_idx_min])}km z3)'   
                   + f' (approx {str("%.0f" % altavg[lev_sl_idx_max,ids])}km : {str("%.0f" % altavg[lev_sl_idx_min,ids])}km alt)'   )


        for it2 in time_ar_2wk:  #loop through two two-week periods each month
            #===================================================================================================
            loop1_start_time = time.process_time()   #loop timing for printout
            #===================================================================================================

            #Select time indices for current loop iteration
            times_idx_min = time_ind_2wk_min[it2]
            times_idx_max = time_ind_2wk_max[it2]

            # Generate time strings for figure labelling
            if ds_months_shape>1:
                times_str_min[ids][it2] = str( times[times_idx_min] ) #~~#
                times_str_max[ids][it2] = str( times[times_idx_max] ) #~~#
                print( '      Time slice ' + str(it2+1) + ' = ' +times_str_min[ids][it2] + ' : ' + times_str_max[ids][it2] ) 
            else:
                times_str_min[it2] = str( times[times_idx_min] ) #~~#
                times_str_max[it2] = str( times[times_idx_max] ) #~~#
                print( '      Time slice ' + str(it2+1) + ' = ' + str( times_str_min[it2] ) + ' : ' + str( times_str_max[it2] ) ) 
            #===================================================================================================

            # Slice metal arrays by chosen alt range and time range 
            tempe = temp[lev_sl_idx_min:lev_sl_idx_max+1,times_idx_min:times_idx_max+1,:,:]
            
            # Mp_t = Mpt[lev_sl_idx_min:lev_sl_idx_max+1,times_idx_min:times_idx_max+1,:,:]
            Fep_t = Fept[lev_sl_idx_min:lev_sl_idx_max+1,times_idx_min:times_idx_max+1,:,:]
            Mgp_t = Mgpt[lev_sl_idx_min:lev_sl_idx_max+1,times_idx_min:times_idx_max+1,:,:]
            Nap_t = Napt[lev_sl_idx_min:lev_sl_idx_max+1,times_idx_min:times_idx_max+1,:,:]

            if crit_freq_on==1:
                elec = elect[lev_sl_idx_min:lev_sl_idx_max+1,times_idx_min:times_idx_max+1,:,:]
            #===================================================================================================

            # VMR to number density calculation
            Feptdens = ( Fep_t * 1e-6 * 100 * lev_sl ) / (1.380503e-23 * tempe)
            Mgptdens = ( Mgp_t * 1e-6 * 100 * lev_sl ) / (1.380503e-23 * tempe)
            Naptdens = ( Nap_t * 1e-6 * 100 * lev_sl ) / (1.380503e-23 * tempe)
            
            Mptdens = Feptdens + (2 * Mgptdens ) + Naptdens    #Approx total metal density            
            
            if crit_freq_on==1:
                edens = ( elec * 1e-6 * 100 * lev_sl ) / (1.380503e-23 * tempe)


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            #==================For Lat-LT PLOTS. Shifting Mptdens from lon to local time========================
            #===================================================================================================
            #Iterate over timear (2wk period, hrly data), offset density at each UT by 15 degrees lon
            for it in timear: 
                Mptdens_sh[:,it,:,:,it2,ids] = np.roll(Mptdens[:,it,:,:], offset[it] , axis=2)
                alt_sl_sh[:,it,:,:,it2,ids] = np.roll(alt_sl[:,it,:,:], offset[it] , axis=2)     #(19,744,96,144,2,3)
                
                if crit_freq_on==1:
                    edens_sh[:,it,:,:,it2,ids] = np.roll(edens[:,it,:,:], offset[it] , axis=2)
            #===================================================================================================
            # Calculate average along time axis for 2wk sample
            alt_sl_sh_avg[:,:,:,it2,ids] = np.mean(alt_sl_sh[:,:,:,:,it2,ids],  axis=1)     #(19,744,96,144,2,3)->(19,96,144,2,3)
            alt_sl_avg[:,:,:,it2,ids] = np.mean(alt_sl,  axis=1)  #average of non-shifted (lat,lon) altitude  #(19,744,96,144)->(19,96,144,2,3)
            #===================================================================================================
            # Calculate average of offset densities along time axis for 2wk sample
            Mptdens_avg = np.mean(Mptdens_sh,  axis=1) #-> (25, 96, 144, 2, 3)

            # Calculate std dev of offset densities along time axis
            Mptdens_std = np.std(Mptdens_sh, axis=1) #-> (25, 96, 144, 2, 3)

            # Calculate difference between density from model output and the average density       
            for it in timear:
                Mptdens_diff[:,it,:,:,it2,ids] = Mptdens_sh[:,it,:,:,it2,ids] - Mptdens_avg[:,:,:,it2,ids]      
            #===================================================================================================
            #Work out zonal avg M layer in 5' lat slices (for each 2wk time slice) & max over lev dim
            Mptdens_avv1 = np.mean(Mptdens_avg, axis=2) #avg Mptdens_avg over lon dim -> (25, 96, 2, 3). (equiv to Mptdens avg'd over timestep & lon dims)
            
            for ilev in levar: #interpolate onto newlat grid 1' spacing (180 long)
                Mptdens_avv1_b[ilev,:,it2,ids] = np.interp( newlat, lat, Mptdens_avv1[ilev,:,it2,ids] )  

                for iintlat in intlat_ar: #average into 5' lat slices
                    Mptdens_avv1_b_5d[ilev,iintlat,it2,ids] = np.mean( Mptdens_avv1_b[ilev,(iintlat*5):((iintlat*5)+5),it2,ids] ) #'GLOBAL' AVG AT HEIGHT X IN 5' SLICES (25, 36, 2, 1)

            max_Mptdens_avv1_b_5d[:,it2,ids] = np.amax(Mptdens_avv1_b_5d[:,:,it2,ids], axis=0) # find max over lev dim  #PEAK OF 'GLOBAL' AVG M LAYER IN 5' SLICES (36, 2, 1)
            #===================================================================================================
            #Assigns correct lat slice (dim 36 long) to variable with normal lat axis (96 long) so this can be used in criteria below. Probably a function to do this much better/in a diff way but I wrote it a long time ago....
            Z = 0
            X = -90
            Y = -85
            for ilat in latar:
                if lat[ilat] > Y :
                    X = X + 5
                    Y = Y + 5 
                    Z = Z + 1
                if (lat[ilat]>=X) & (lat[ilat]<=Y) :
                    max_Mptdens_avv1b5d_l[ ilat ,it2,ids] = max_Mptdens_avv1_b_5d[ Z ,it2,ids] #peak of layer over lev dim
                    Mptdens_avv1b5d_l[:, ilat ,it2,ids] = Mptdens_avv1_b_5d[:, Z ,it2,ids]  #metal layer as fct of height       
            #===================================================================================================
            # CRITERIA & Es Identifiation Calculations (LT)
            #===================================================================================================
            # Where criteria set are met, set SpEs array to Mptdens_sh (total metal density), otherwise set to NaN
            #Criteria are: ( Diff > 0.25x sigma )  &  ( Mpt > 2x glb average at height x in 5' slice)  &  ( Mpt > 1x glb avg layer peak in 5' slice)
            
            SpEs[:,:,:,:,it2,ids] = np.where( ( Mptdens_diff[:,:,:,:,it2,ids]>( 0.25 * sigma_val*Mptdens_std[:,None,:,:,it2,ids]) ) & 
                                                ( Mptdens_sh[:,:,:,:,it2,ids] > ( 2 * Mptdens_avv1b5d_l[:,None,:,None,it2,ids] ) ) &    
                                               ( Mptdens_sh[:,:,:,:,it2,ids] > ( 1 * max_Mptdens_avv1b5d_l[None,None,:,None,it2,ids] ) )    , Mptdens_sh[:,:,:,:,it2,ids] , SpEs_sh_nan[:,:,:,:,it2,ids] )
            # 
                
            if crit_freq_on==1: 
                SpEs_e[:,:,:,:,it2,ids] = np.where( SpEs[:,:,:,:,it2,ids]==Mptdens_sh[:,:,:,:,it2,ids], edens_sh[:,:,:,:,it2,ids] , SpEs_sh_nan[:,:,:,:,it2,ids] ) 

            
            #===================================================================================================
            if crit_freq_on==1:    
                #Calculate critical ionosonde frequency    
                maxSpEs__e[:,:,:,it2,ids] = np.nanmax(SpEs_e[:,:,:,:,it2,ids], axis=0)  #find max over lev dim
                maxSpEs_e[:,:,:,it2,ids] =maxSpEs__e[:,:,:,it2,ids] * 1e6   #cm-3 -> m-3

                foEs__m[:,:,:,it2,ids] = 8.98 * np.sqrt(maxSpEs_e[:,:,:,it2,ids])
                foEs_m[:,:,:,it2,ids] = foEs__m[:,:,:,it2,ids] / 1e6    #Hz -> MHz

                foEs_m_av[:,:,it2,ids] = np.nanmean( foEs_m[:,:,:,it2,ids], axis=0 ) #avg over lev dim

            #===================================================================================================
            #Calculate Occurence freq
            SpEs_freq[:,:,:,:,it2,ids] = np.isfinite(SpEs[:,:,:,:,it2,ids]) *1.  #Where SpEs is a number, set to True, otherwise set to False, then convert True/False to 1/0s
            SpEs_freq_time[:,:,:,it2,ids] = np.sum(SpEs_freq[:,:,:,:,it2,ids], axis=1) #Sum over Time dim to give SpEs_freq_time (total occurences in 2 week time period) -> (19,96,144,2,3)
            SpEs_Occ_Freq[:,:,:,it2,ids] = ( SpEs_freq_time[:,:,:,it2,ids] / time_shape ) *100.    #Divide by number of timesteps (336) to give occurence freq (%)  # (19,96,144,2,3)
            
            #interpolate from x144 inds to xLTshape
            for ilev in levar:
                for ilat in latar:
                    SpEs_Occ_Fr_b[ilev,ilat,:,it2,ids] = np.interp(LTar, LTlong, SpEs_Occ_Freq[ilev,ilat,:,it2,ids])
                    Mptdens_avg_b[ilev,ilat,:,it2,ids] = np.interp(LTar, LTlong, Mptdens_avg[ilev,ilat,:,it2,ids])  
                    alt_sl_sh_avg_b[ilev,ilat,:,it2,ids] = np.interp(LTar, LTlong, alt_sl_sh_avg[ilev,ilat,:,it2,ids])
                    
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # For alt-LT plots (want to plot an average for a 5' lat slice)
            # Bin into 1' lat bins instead of 1.89' (180 long) then avg in 5' slices   
            for ilev in levar:
                for iit_arr in it_arr: #interpolate onto newlat grid 1' spacing (180 long) (was 1.89')
                    SpEs_Occ_Fr_bb[ilev,:,iit_arr,it2,ids] = np.interp( newlat, lat, SpEs_Occ_Fr_b[ilev,:,iit_arr,it2,ids] )  
                    Mptdens_avg_bb[ilev,:,iit_arr,it2,ids] = np.interp( newlat, lat, Mptdens_avg_b[ilev,:,iit_arr,it2,ids] )  
                    alt_sl_sh_avg_bb[ilev,:,iit_arr,it2,ids] = np.interp( newlat, lat, alt_sl_sh_avg_b[ilev,:,iit_arr,it2,ids] )  

                    for iintlat in intlat_ar: #avg interpolated array into 5' slices
                        SpEs_Occ_Fr_bb_5d[ilev,iintlat,iit_arr,it2,ids] = np.mean(   SpEs_Occ_Fr_bb[ilev,(iintlat*5):((iintlat*5)+5),iit_arr,it2,ids] )   #average in 5' slices
                        Mptdens_avg_bb_5d[ilev,iintlat,iit_arr,it2,ids] = np.mean(   Mptdens_avg_bb[ilev,(iintlat*5):((iintlat*5)+5),iit_arr,it2,ids] )   #average in 5' slices
                        alt_sl_sh_avg_bb_5d[ilev,iintlat,iit_arr,it2,ids] = np.mean(   alt_sl_sh_avg_bb[ilev,(iintlat*5):((iintlat*5)+5),iit_arr,it2,ids] )   #average in 5' slices

            #=================================================================================================== 
            #Calculate Occurrence freq over lat-LT by summing over alt dim (for lat-LT plots)
            SpEs_freq_altsum[:,:,:,it2,ids] = np.sum(SpEs_freq[:,:,:,:,it2,ids],axis=0) #sum over alt dim
            SpEs_freq_altsum_time[:,:,it2,ids] = np.sum(SpEs_freq_altsum[:,:,:,it2,ids],axis=0)  #sum over time dim
            SpEs_Occ_Freq_ll[:,:,it2,ids] = SpEs_freq_altsum_time[:,:,it2,ids] / (time_shape*lev_shape) * 100.
            #- - - - - - - - - - - - - - - - - - - - - - - - - - -
            #interpolate from x144 LT inds to xLTshape
            for ilat in latar:
                SpEs_Occ_Freq_llb[ilat,:,it2,ids] = np.interp(LTar, LTlong, SpEs_Occ_Freq_ll[ilat,:,it2,ids]) 
            
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            #====================For Lat-LON PLOTS. Repeat of above usng non-shifted Mptdens====================
            #===================================================================================================

            Mptdensns[:,:,:,:,it2,ids] = Mptdens[:,:,:,:] #(19, 336, 96, 144, 2, 3)
            if crit_freq_on==1:
                edensns[:,:,:,:,it2,ids] = edens[:,:,:,:] #(19, 336, 96, 144, 2, 3)

            # Calculate average of densities along time axis for 2wk sample
            Mptdens_nsavg[:,:,:,it2,ids] = np.mean(Mptdensns[:,:,:,:,it2,ids],  axis=1) #-> (19, 96, 144, 2, 3)

            # Calculate std dev of densities along time axis
            Mptdens_nsstd[:,:,:,it2,ids] = np.std(Mptdensns[:,:,:,:,it2,ids], axis=1) #-> (19, 96, 144, 2, 3)

            # Calculate difference between density from model output and the average density     
            for it in timear:
                Mptdens_nsdiff[:,it,:,:,it2,ids] = Mptdensns[:,it,:,:,it2,ids] - Mptdens_nsavg[:,:,:,it2,ids]      
            #===================================================================================================
            #Work out zonal avg M layer in 5' lat slices (for each 2wk time slice) & max over lev dim
            Mptdens_nsavv1 = np.mean(Mptdens_nsavg, axis=2) #avg Mptdens_avg over lon dim -> (19, 96, 2, 3). (equiv to Mptdens avg'd over timestep & lon dims)
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - 
            for ilev in levar: #interpolate onto newlat grid 1' spacing (180 long)
                Mptdens_nsavv1_b[ilev,:,it2,ids] = np.interp( newlat, lat, Mptdens_nsavv1[ilev,:,it2,ids] )  

                for iintlat in intlat_ar: #average into 5' lat slices
                    Mptdens_nsavv1_b_5d[ilev,iintlat,it2,ids] = np.mean( Mptdens_nsavv1_b[ilev,(iintlat*5):((iintlat*5)+5),it2,ids] ) #'GLOBAL' AVG AT HEIGHT X IN 5' SLICES (19, 36, 2, 1)

            max_Mptdens_nsavv1_b_5d[:,it2,ids] = np.amax(Mptdens_nsavv1_b_5d[:,:,it2,ids], axis=0) # find max over lev dim  #PEAK OF 'GLOBAL' AVG M LAYER IN 5' SLICES (36, 2, 1)
            #===================================================================================================
            #Assigns correct lat slice (dim 36 long) to variable with normal lat axis (96 long) so this can be used in criteria below
            Z = 0
            X = -90
            Y = -85
            for ilat in latar:
                if lat[ilat] > Y :
                    X = X + 5
                    Y = Y + 5 
                    Z = Z + 1
                if (lat[ilat]>=X) & (lat[ilat]<=Y) :
                    max_Mptdens_nsavv1b5d_l[ ilat ,it2,ids] = max_Mptdens_nsavv1_b_5d[ Z ,it2,ids] #peak of layer over lev dim
                    Mptdens_nsavv1b5d_l[:, ilat ,it2,ids] = Mptdens_nsavv1_b_5d[:, Z ,it2,ids]  #metal layer as fct of height  
                    
            #===================================================================================================
            # CRITERIA & Es Identifiation Calculations (Lon)
            #===================================================================================================
            # Where criteria set are met, set SpEs to Mptdens, otherwise set to NaN
            #Criteria are: ( Diff > 0.25x sigma )  &  ( Mpt > 2x glb average at height x in 5' slice)  &  ( Mpt > 1x glb avg layer peak in 5' slice)   

            SpEsns[:,:,:,:,it2,ids] = np.where( ( Mptdens_nsdiff[:,:,:,:,it2,ids]>( 0.25 * sigma_val*Mptdens_nsstd[:,None,:,:,it2,ids]) ) &  
                                                   ( Mptdensns[:,:,:,:,it2,ids] > ( 2 * Mptdens_nsavv1b5d_l[:,None,:,None,it2,ids] ) ) &  
                                                 ( Mptdensns[:,:,:,:,it2,ids] > ( 1 * max_Mptdens_nsavv1b5d_l[None,None,:,None,it2,ids] ) ) , Mptdensns[:,:,:,:,it2,ids] , SpEs_sh_nan[:,:,:,:,it2,ids] )
            #
            
            if crit_freq_on==1: 
                SpEs_nse[:,:,:,:,it2,ids] = np.where( SpEsns[:,:,:,:,it2,ids]==Mptdensns[:,:,:,:,it2,ids], edensns[:,:,:,:,it2,ids] , SpEs_sh_nan[:,:,:,:,it2,ids] ) 

            #===================================================================================================
            if crit_freq_on==1:    
                #Calculate critical ionosonde frequency    
                maxSpEs__nse[:,:,:,it2,ids] = np.nanmax(SpEs_nse[:,:,:,:,it2,ids], axis=0)  #find max over lev dim
                maxSpEs_nse[:,:,:,it2,ids] =maxSpEs__nse[:,:,:,it2,ids] * 1e6   #cm-3 -> m-3

                foEsns__m[:,:,:,it2,ids] = 8.98 * np.sqrt(maxSpEs_nse[:,:,:,it2,ids])
                foEsns_m[:,:,:,it2,ids] = foEsns__m[:,:,:,it2,ids] / 1e6    #Hz -> MHz

                foEsns_m_av[:,:,it2,ids] = np.nanmean( foEsns_m[:,:,:,it2,ids], axis=0 ) #avg over timestep dim

            #===================================================================================================
            #Calculate Occurence freq

            SpEsns_freq[:,:,:,:,it2,ids] = np.isfinite(SpEsns[:,:,:,:,it2,ids]) *1. #Convert True/False to 1/0s -> (19,336,96,144,2,3)
            SpEsns_freq_time[:,:,:,it2,ids] = np.sum(SpEsns_freq[:,:,:,:,it2,ids], axis=1) #Sum over Time dim to give SpEs_freq_time (total occurences in 2 week time period) -> (19,96,144,2,3)
            SpEsns_Occ_Freq[:,:,:,it2,ids] = ( SpEsns_freq_time[:,:,:,it2,ids] / time_shape ) *100.    #Divide by number of timesteps (336) to give occurence freq (%)  # (19,96,144,2,3)
            
            SpEsns_freq_altsum[:,:,:,it2,ids] = np.sum(SpEsns_freq[:,:,:,:,it2,ids],axis=0) #sum over alt dim
            SpEsns_freq_altsum_time[:,:,it2,ids] = np.sum(SpEsns_freq_altsum[:,:,:,it2,ids],axis=0)  #sum over time dim
            SpEsns_Occ_Freq_ll[:,:,it2,ids] = SpEsns_freq_altsum_time[:,:,it2,ids] / (time_shape*lev_shape) * 100.
            
            
            #===================================================================================================
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////   


            #===================================================================================================
            #Calculate loop timings for printout
            loop1_end_time = time.process_time()
            loop1_time_taken = loop1_end_time - loop1_start_time
            loop1_time_taken_min = loop1_time_taken / 60.

            print('      Time slice ' + str(it2+1) + ' = ' + str(loop1_time_taken_min) + ' mins' ) 
            print('      ----------------------------------------')
            #===================================================================================================




        #--- Out of it2 loop (2wk time periods), still in ds loop (month) --- Calculating monthly averages for relevant variables, including altitude with same dims as each output array
 
        ###__SpEs_Occ_Fr__###
    
        #Lat,LT at specific heights, monthly avg -> (lev,lat,LT,ids)
        #(Using SpEs_Occ_Fr_b averaged in 0.5h LT slices but not into 5' lat slices)
        SpEs_Occ_Fr_b_avg[:,:,:,ids] = np.mean(SpEs_Occ_Fr_b[:,:,:,:,ids],  axis=3) #avg over it2 dim (avg both 2wk periods)  => Monthly avg (19, 96, 48, 3)
        alt_sl_sh_avg_b_avg[:,:,:,ids] = np.mean(alt_sl_sh_avg_b[:,:,:,:,ids],  axis=3)  #(19,96,48,2,3)->(19,96,48,3)
        
        #LT avg ->(lev,lat,ids)   
        SpEs_Occ_Fr_b_avgLT[:,:,ids] = np.mean(SpEs_Occ_Fr_b_avg[:,:,:,ids],  axis=2)  #avg over LT dim =>  #alt-lat monthly avg (19, 96, 3) 
        alt_sl_sh_avg_b_avgLT[:,:,ids] = np.mean(alt_sl_sh_avg_b_avg[:,:,:,ids],  axis=2)  #avg over LT dim =>  #alt-lat monthly avg (19, 96, 3) 

        #5' lat bins ->(lev,5'lat,LT,ids) 
        SpEs_Occ_Fr_bb_5d_avg[:,:,:,ids] = np.mean(SpEs_Occ_Fr_bb_5d[:,:,:,:,ids],  axis=3) #avg over it2 dim (avg both 2wk periods) => Monthly avg   ->(19, 36, 48, 3)
        alt_sl_sh_avg_bb_5d_avg[:,:,:,ids] = np.mean(alt_sl_sh_avg_bb_5d[:,:,:,:,ids],  axis=3) 

        #Lat-Lon ->(lev,lat,lon,ids)
        SpEsns_Occ_Fr_avg[:,:,:,ids] = np.mean(SpEsns_Occ_Freq[:,:,:,:,ids],  axis=3) #avg over it2 dim (avg both 2wk periods)  => Monthly avg (19,96,144,3)
        alt_sl_aavg[:,:,:,ids] = np.mean(alt_sl_avg[:,:,:,:,ids],  axis=3)     #lon axis   #(19,96,144,2,3)->(19,96,144,3)

        SpEs_Occ_Freq_llba[:,:,ids] = np.mean(SpEs_Occ_Freq_llb[:,:,:,ids],  axis=2)  #average over it2 dim 
        SpEsns_Occ_Freq_lla[:,:,ids] = np.mean(SpEsns_Occ_Freq_ll[:,:,:,ids],  axis=2)
        
        
        ###__FoEs_Monthly_avg___###
        if crit_freq_on==1:
            foEs_m_av_mth[:,:,ids] = np.nanmean( foEs_m_av[:,:,:,ids], axis=2 )
            foEsns_m_av_mth[:,:,ids] = np.nanmean( foEsns_m_av[:,:,:,ids], axis=2 )



        #===================================================================================================
        #Loop timings for printout
        loop_end_time = time.process_time()
        loop_time_taken = loop_end_time - loop_start_time
        loop_time_taken_min = loop_time_taken / 60.

        print('         Month ' + str(ds_months[ids]) + ' Time = ' + str(loop_time_taken_min) + ' mins' )  
        print('========================================')
        #===================================================================================================



    #--- Out of it2 loop (2wk time periods) AND ds loop (month) ---   

    ###__SpEs_Occ_Fr__###
    
    #(lev,lat,LT), Lat,LT at specific heights 
    SpEs_Occ_Fr_b_dsavg[:,:,:] = np.mean(SpEs_Occ_Fr_b_avg[:,:,:,:],  axis=3) #avg over ids dim (avg all months)   =Whole Dataset avg   ->(19, 96, 48)
    alt_sl_sh_avg_b_dsavg[:,:,:] = np.mean(alt_sl_sh_avg_b_avg[:,:,:,:],  axis=3)
 
    #(lev,lat) LT avg 
    SpEs_Occ_Fr_b_dsavgLT[:,:] = np.mean(SpEs_Occ_Fr_b_avgLT[:,:,:],  axis=2) #avg over ids dim ->(19, 96)
    alt_sl_sh_avg_b_dsavgLT[:,:] = np.mean(alt_sl_sh_avg_b_avgLT[:,:,:],  axis=2) #avg over ids dim ->(25, 96)

    #(lev,5'lat,LT), 5' lat bins 
    SpEs_Occ_Fr_bb_5d_dsavg[:,:,:] = np.mean(SpEs_Occ_Fr_bb_5d_avg[:,:,:,:],  axis=3) #avg over ids dim (avg all months)   =Whole Dataset avg   ->(19, 36, 48)
    alt_sl_sh_avg_bb_5d_dsavg[:,:,:] = np.mean(alt_sl_sh_avg_bb_5d_avg[:,:,:,:],  axis=3)
    
    # (lev,lat,lon) 
    SpEsns_Occ_Fr_dsavg[:,:,:] = np.mean(SpEsns_Occ_Fr_avg[:,:,:,:],  axis=3) #avg over ids dim (avg all months)  => Whole Dataset avg (19,96,144)
    alt_sl_aaavg[:,:,:] = np.mean(alt_sl_aavg,  axis=3)     #(19,96,144,3)->(19,96,144)

    altavg = np.mean(altavg,  axis=1)
    altavg_sl = np.mean(altavg_sl,  axis=1)
    
    # (lat,LT) 
    SpEs_Occ_Freq_llbav[:,:] = np.mean(SpEs_Occ_Freq_llba[:,:,:],  axis=2)    #avg over ids dim => (lat,LT)
    # (lat,lon)
    SpEsns_Occ_Freq_llav[:,:] = np.mean(SpEsns_Occ_Freq_lla[:,:,:],  axis=2)  #avg over ids dim => (lat,lon)
    # (lat)
    SpEsns_Occ_Freq_lat = np.mean(SpEsns_Occ_Freq_llav[:,:], axis=1)
    
    
    ###__FoEs_ds_avg___###
    if crit_freq_on==1:
        foEs_m_av_ds[:,:] = np.nanmean( foEs_m_av_mth[:,:,:], axis=2 )
        foEsns_m_av_ds[:,:] = np.nanmean( foEsns_m_av_mth[:,:,:], axis=2 )

    
    #===================================================================================================
    end_time = time.process_time()
    time_taken = end_time - start_time
    time_taken_min = time_taken / 60.
    print('========================================')
    print('Calculation Time = ' + str(time_taken_min) + ' mins' )  

    
    #Save results to nc file
    save_results(run_name, Monthfolderstr, lev, lev_sl, timear, lat, intlat, lon, LTar, LTlong, time_ar_2wk, ds_months, Zavg_sl, altavg, altavg_sl, times_str_min, times_str_max, SpEs_Occ_Fr_b_dsavg, SpEs_Occ_Fr_b_avg, SpEs_Occ_Fr_bb_5d_dsavg, SpEsns_Occ_Fr_dsavg, SpEs, alt_sl_sh_avg_b_dsavg, alt_sl_sh_avg_b_dsavgLT, alt_sl_sh_avg_bb_5d_dsavg, Mptdens_avv1_b_5d, Mptdens_std, Mptdens_nsstd, SpEsns_freq_time, SpEs_freq_time, Mptdens_avg, Mptdens_nsdiff, Mptdens_nsavg, SpEs_Occ_Freq_llbav, SpEsns_Occ_Freq_llav, SpEsns_Occ_Freq_lat  )
    
    


    
# -




