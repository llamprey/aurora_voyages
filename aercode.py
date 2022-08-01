import matplotlib 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import xarray as xr

# some constants
Rd=287.05
cp=1005.46 # J/kg/K 1.0035
Rd_cp = Rd / cp
p0=1.0e5 # Pa
zboltz=1.3807e-23
avc=6.022e23
mm_da=avc*zboltz/Rd #molecular mass of dry air

# Molecular masses
# n.b. mm_bc=0.012, mm_oc=mm_so=0.0168=1.4*mm_c. assume both primary & secondary organic species have mm=1.4*mm_c
mm_aer=[0.098,0.012,0.0168,0.05844,0.100,0.0168] 
cp_so4=0
cp_bc=1
cp_oc=2
cp_nacl=3
cp_du=4


def keycutter(f_path, time, lat, lon):
    """Converts IMOS-SOOP-Air Sea Flux (ASF) sub-facility-Meteorological 
    and SST Observations data into a 'location key' that can be used to 
    extract model data along a ship track.
    
    Args:
      path (str): Underway data file location
      time (str): Name of datetime column
      lat (str): Name of latitude column
      lon (str): Name of longitude column
      
      Returns a 'key' with 'time', 'lat', and 'lon' data columns
    """
    
    uw = pd.read_csv(f_path, delimiter=',', skiprows=54, index_col=time)
    uw[time] = pd.DatetimeIndex(uw.index)
    uw.index = uw[time].astype('datetime64[ns]')
    uw = uw[[lat,lon]].rename(columns={lat: 'lat', lon: 'lon'})
    uw.index.names = ['time']
    uw = uw.resample('1D', kind='time').mean()
    return uw


def df_md(key, m_path, mod, dt):
    """Uses position data to extract model data for that position.
    
    Args:
      key (Dataframe): must contain 'time', 'lat', and 'lon' data columns. See keycutter function.
      m_path (str): filepath for the model data.
      mod (str): name of the model run identifier.
      dt (str): type of data, either paer, pche or pmet.
    
    Returns a data array containing model data set by 'dt'.
    """
        
    for t in key.index: #loop to pull out model data on ship track
        m_df = m_path+mod+'.'+dt+'{}{:02d}{:02d}.nc'.format(t.year,t.month,t.day)
        df = xr.open_mfdataset(m_df)
        shplat = key['lat'].loc[t]
        shplon = key['lon'].loc[t]
        data = df.interp(lat=shplat, lon=shplon, lat_v=shplat, lon_u=shplon)
        data = data.reset_coords(('lat','lon','lat_v','lon_u'))
        data['lat'] = data.lat.expand_dims(dim='time')
        data['lon'] = data.lon.expand_dims(dim='time')
        data['lat_v'] = data.lat_v.expand_dims(dim='time')
        data['lon_u'] = data.lon_u.expand_dims(dim='time')        
        if t == key.index[0]:
            dftrack = data
        else:
            dftrack = xr.concat([data, dftrack], dim='time')
    return dftrack


def calc_density(met): # Need in the met file, theta and pressure on theta levels
    tempk = met.theta*((met.field408/p0)**Rd_cp) # Calculate temp 
    density = (met.field408/(tempk*zboltz*1.0E6)) # Calculate number density
    met = xr.merge([met,{'tempk':tempk,'density':density}]) # add them to array
    return met


def calc_density2(met): # Need in the met file, tempk, theta and pressure on theta levels
    density = (met.field408/(met.ta*zboltz*1.0E6)) # Calculate number density
    met = xr.merge([met,{'density':density}]) # add them to array
    return met


# Convert aerosol/gas phases into more useful units (from volume/mass mixing ratios)
# Need density in the met file, and all the aerosol fields in the aer file 
def aero_unit_conversions(aer,met):
    
    #MMRs
    aer['field34071'] = aer['field34071']*((29./62.) * 1e9) # DMS 
    aer['field34071'] = aer['field34071'].assign_attrs({'units':'ppb'})

    aer['field34072'] = aer['field34072']*((29./64.) * 1e9) # SO2
    aer['field34072'] = aer['field34072'].assign_attrs({'units':'ppb'})

    aer['field34073'] = aer['field34073']*((29./98.) * 1e12) # gas phase H2SO4
    aer['field34073'] = aer['field34073'].assign_attrs({'units':'ppt'})

    # Number densities 
    for nd in ['field34101','field34103','field34107','field34113','field34119']:
        aer[nd] = aer[nd] * met.density
        aer[nd] = aer[nd].assign_attrs({'units':'cm-3'})
    
    # Mass desnsities
    # H2SO4   
    for md in ['field34102','field34104','field34108','field34114']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_so4] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Organics
    for md in ['field34126','field34106','field34121','field34110','field34116']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_oc] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Sea salt
    for md in ['field34111','field34117']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_nacl] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Black Carbon
    for md in ['field34105','field34109','field34115','field34120']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_bc] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Dust 
    for md in ['field431','field432','field433','field434','field435','field436']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_du] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})    
        
    return aer


# Convert aerosol/gas phases into more useful units (from volume/mass mixing ratios)
# Need density in the met file, and all the aerosol fields in the aer file
# USE if DMS, SO2 and gas phase H2SO4 are in che file, not aer file
def aero_unit_conversions2(aer,met):
    
    
    # Number densities 
    for nd in ['field34101','field34103','field34107','field34113','field34119']:
        aer[nd] = aer[nd] * met.density
        aer[nd] = aer[nd].assign_attrs({'units':'cm-3'})
    
    # Mass desnsities
    # H2SO4   
    for md in ['field34102','field34104','field34108','field34114']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_so4] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Organics
    for md in ['field34126','field34106','field34121','field34110','field34116']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_oc] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Sea salt
    for md in ['field34111','field34117']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_nacl] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Black Carbon
    for md in ['field34105','field34109','field34115','field34120']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_bc] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Dust 
    for md in ['field431','field432','field433','field434','field435','field436']:
        aer[md] = aer[md] * (mm_da/mm_aer[cp_du] * met.density/avc)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})    
        
    return aer


# Convert mass mixing ratios to more useful units
def chem_unit_conversions(che):

    che['field34001'] = che['field34001']*((29./48.) * 1e9) # O3
    che['field34001'] = che['field34001'].assign_attrs({'units':'ppb'})
    che['field34001'] = che['field34001'].assign_attrs({'long_name':'Ozone'})

    che['field34002'] = che['field34002']*((29./30.) * 1e9) # NO
    che['field34002'] = che['field34002'].assign_attrs({'units':'ppb'})
    che['field34002'] = che['field34002'].assign_attrs({'long_name':'Nitric Oxide'})

    che['field34003'] = che['field34003']*((29./62.) * 1e9) # NO3
    che['field34003'] = che['field34003'].assign_attrs({'units':'ppb'})
    che['field34003'] = che['field34003'].assign_attrs({'long_name':'Nitrate'})

    che['field34005'] = che['field34005']*((29./108.) * 1e9) # N2O5
    che['field34005'] = che['field34005'].assign_attrs({'units':'ppb'})
    che['field34005'] = che['field34005'].assign_attrs({'long_name':'Dinitrogen Pentoxide'})

    che['field34006'] = che['field34006']*((29./79.) * 1e9) # HO2NO2
    che['field34006'] = che['field34006'].assign_attrs({'units':'ppb'})
    che['field34006'] = che['field34006'].assign_attrs({'long_name':'Peroxynitric Acid'})    

    che['field34007'] = che['field34007']*((29./63.) * 1e9) # NONO2
    che['field34007'] = che['field34007'].assign_attrs({'units':'ppb'})
    che['field34007'] = che['field34007'].assign_attrs({'long_name':'Nitric Acid'})

    che['field34008'] = che['field34008']*((29./34.) * 1e9) # H2O2
    che['field34008'] = che['field34008'].assign_attrs({'units':'ppb'})
    che['field34008'] = che['field34008'].assign_attrs({'long_name':'Hygrogen Peroxide'})

    che['field34009'] = che['field34009']*((29./16.) * 1e9) # CH4
    che['field34009'] = che['field34009'].assign_attrs({'units':'ppb'})
    che['field34009'] = che['field34009'].assign_attrs({'long_name':'Methane'})

    che['field34010'] = che['field34010']*((29./28.) * 1e9) # CO
    che['field34010'] = che['field34010'].assign_attrs({'units':'ppb'})
    che['field34010'] = che['field34010'].assign_attrs({'long_name':'Carbon Monoxide'})

    che['field34011'] = che['field34011']*((29./30.) * 1e9) # HCHO
    che['field34011'] = che['field34011'].assign_attrs({'units':'ppb'})
    che['field34011'] = che['field34011'].assign_attrs({'long_name':'Formaldehyde'})

    che['field34013'] = che['field34013']*((29./47.) * 1e9) # HONO
    che['field34013'] = che['field34013'].assign_attrs({'units':'ppb'})
    che['field34013'] = che['field34013'].assign_attrs({'long_name':'Nitrous Acid'})

    che['field34041'] = che['field34041']*((29./35.) * 1e9) # Cl
    che['field34041'] = che['field34041'].assign_attrs({'units':'ppb'})
    che['field34041'] = che['field34041'].assign_attrs({'long_name':'Chlorine'})

    che['field34042'] = che['field34042']*((29./51.) * 1e9) # ClO
    che['field34042'] = che['field34042'].assign_attrs({'units':'ppb'})
    che['field34042'] = che['field34042'].assign_attrs({'long_name':'Hypochlorite'})

    che['field34043'] = che['field34043']*((29./103.) * 1e9) # Cl2O2
    che['field34043'] = che['field34043'].assign_attrs({'units':'ppb'})
    che['field34043'] = che['field34043'].assign_attrs({'long_name':'Chlorine Peroxide'})

    che['field34044'] = che['field34044']*((29./67.) * 1e9) # OClO
    che['field34044'] = che['field34044'].assign_attrs({'units':'ppb'})
    che['field34044'] = che['field34044'].assign_attrs({'long_name':'Chlorine Superoxide'})

    che['field34045'] = che['field34045']*((29./80.) * 1e9) # Br
    che['field34045'] = che['field34045'].assign_attrs({'units':'ppb'})
    che['field34045'] = che['field34045'].assign_attrs({'long_name':'Bromine'})

    che['field34047'] = che['field34047']*((29./115.) * 1e9) # BrCl
    che['field34047'] = che['field34047'].assign_attrs({'units':'ppb'})
    che['field34047'] = che['field34047'].assign_attrs({'long_name':'Bromine Chloride'})

    che['field34048'] = che['field34048']*((29./142.) * 1e9) #BrONO2
    che['field34048'] = che['field34048'].assign_attrs({'units':'ppb'})
    che['field34048'] = che['field34048'].assign_attrs({'long_name':'Bromine Nitrate'})

    che['field34049'] = che['field34049']*((29./44.) * 1e9) # N2O
    che['field34049'] = che['field34049'].assign_attrs({'units':'ppb'})
    che['field34049'] = che['field34049'].assign_attrs({'long_name':'Nitrous Oxide'})

    che['field34051'] = che['field34051']*((29./52.) * 1e9) # HOCl
    che['field34051'] = che['field34051'].assign_attrs({'units':'ppb'})
    che['field34051'] = che['field34051'].assign_attrs({'long_name':'Hypochlorous Acid'})

    che['field34052'] = che['field34052']*((29./81.) * 1e9) # HBr
    che['field34052'] = che['field34052'].assign_attrs({'units':'ppb'})
    che['field34052'] = che['field34052'].assign_attrs({'long_name':'Hydrogen Bromide'})

    che['field34053'] = che['field34053']*((29./97.) * 1e9) # HOBr
    che['field34053'] = che['field34053'].assign_attrs({'units':'ppb'})
    che['field34053'] = che['field34053'].assign_attrs({'long_name':'Hypoborous Acid'})

    che['field34054'] = che['field34054']*((29./97.) * 1e9) # ClONO2
    che['field34054'] = che['field34054'].assign_attrs({'units':'ppb'})
    che['field34054'] = che['field34054'].assign_attrs({'long_name':'Chlorine Nitrate'})

    che['field34055'] = che['field34055']*((29./137.) * 1e9) # CFCl3
    che['field34055'] = che['field34055'].assign_attrs({'units':'ppb'})
    che['field34055'] = che['field34055'].assign_attrs({'long_name':'Trichlorofluromethane'})

    che['field34056'] = che['field34056']*((29./121.) * 1e9) # CF2Cl2
    che['field34056'] = che['field34056'].assign_attrs({'units':'ppb'})
    che['field34056'] = che['field34056'].assign_attrs({'long_name':'Dichlorodifluromethane'})

    che['field34057'] = che['field34057']*((29./95.) * 1e9) # MeBr
    che['field34057'] = che['field34057'].assign_attrs({'units':'ppb'})
    che['field34057'] = che['field34057'].assign_attrs({'long_name':'Methyl Bromide'})

    che['field34070'] = che['field34070']*((29./2.) * 1e9) # H2
    che['field34070'] = che['field34070'].assign_attrs({'units':'ppb'})
    che['field34070'] = che['field34070'].assign_attrs({'long_name':'Hydrogen Gas'})

    che['field34071'] = che['field34071']*((29./62.) * 1e9) # DMS 
    che['field34071'] = che['field34071'].assign_attrs({'units':'ppb'})
    che['field34071'] = che['field34071'].assign_attrs({'long_name':'Dimethyl Sulfide'})

    che['field34072'] = che['field34072']*((29./64.) * 1e9) # SO2
    che['field34072'] = che['field34072'].assign_attrs({'units':'ppb'})
    che['field34072'] = che['field34072'].assign_attrs({'long_name':'Sulfur Dioxide'})

    che['field34073'] = che['field34073']*((29./98.) * 1e12) # gas phase H2SO4
    che['field34073'] = che['field34073'].assign_attrs({'units':'ppt'})
    che['field34073'] = che['field34073'].assign_attrs({'long_name':'Sulfuric Acid'})

    che['field34074'] = che['field34074']*((29./96.) * 1e9) # MSA
    che['field34074'] = che['field34074'].assign_attrs({'units':'ppb'})
    che['field34074'] = che['field34074'].assign_attrs({'long_name':'Methylsulfonic Acid'})

    che['field34075'] = che['field34075']*((29./78.) * 1e9) # DMSO
    che['field34075'] = che['field34075'].assign_attrs({'units':'ppb'})
    che['field34075'] = che['field34075'].assign_attrs({'long_name':'Dimethyl Sulfoxide'})

    che['field34076'] = che['field34076']*((29./17.) * 1e9) # NH3
    che['field34076'] = che['field34076'].assign_attrs({'units':'ppb'})
    che['field34076'] = che['field34076'].assign_attrs({'long_name':'Ammonia'})

    che['field34077'] = che['field34077']*((29./76.) * 1e9) # CS2
    che['field34077'] = che['field34077'].assign_attrs({'units':'ppb'})
    che['field34077'] = che['field34077'].assign_attrs({'long_name':'Carbon Sulfide'})

    che['field34078'] = che['field34078']*((29./60.) * 1e9) # COS
    che['field34078'] = che['field34078'].assign_attrs({'units':'ppb'})
    che['field34078'] = che['field34078'].assign_attrs({'long_name':'Carbonyl Sulfide'})

    che['field34079'] = che['field34079']*((29./34.) * 1e9) # H2S
    che['field34079'] = che['field34079'].assign_attrs({'units':'ppb'})
    che['field34079'] = che['field34079'].assign_attrs({'long_name':'Hydrogen Sulfide'})

    che['field34081'] = che['field34081']*((29./17.) * 1e9) # OH
    che['field34081'] = che['field34081'].assign_attrs({'units':'ppb'})
    che['field34081'] = che['field34081'].assign_attrs({'long_name':'Hydroxide'})

    che['field34082'] = che['field34082']*((29./33.) * 1e9) # HO2
    che['field34082'] = che['field34082'].assign_attrs({'units':'ppb'})
    che['field34082'] = che['field34082'].assign_attrs({'long_name':'Hydroperoxyl'})

    che['field34992'] = che['field34992']*((29./36.) * 1e9) # HCl
    che['field34992'] = che['field34992'].assign_attrs({'units':'ppb'})
    che['field34992'] = che['field34992'].assign_attrs({'long_name':'Hydrogen Chloride'})

    che['field34994'] = che['field34994']*((29./96.) * 1e9) # BrO
    che['field34994'] = che['field34994'].assign_attrs({'units':'ppb'})
    che['field34994'] = che['field34994'].assign_attrs({'long_name':'Hypobromite'})

    che['field34996'] = che['field34996']*((29./46.) * 1e9) # NO2
    che['field34996'] = che['field34996'].assign_attrs({'units':'ppb'})
    che['field34996'] = che['field34996'].assign_attrs({'long_name':'Nitrogen Dioxide'})
    
    return che


# function to calculate the cumulative total number to a given radius of a given input mode
def lognormal_cumulative_to_r(N,r,rbar,sigma_g):

    total_to_r=(N/2.0)*(1.0+scipy.special.erf(np.log(r/rbar)/np.sqrt(2.0)/np.log(sigma_g)))

    return total_to_r


# +
# calculates N3 and N10 number concentrations from aerosol model data

def nt_calcs(aer):
    nsteps = len(aer.time)
    nmodes = 7
    nlevels=len(aer.z1_hybrid_height)
    nlats = len(aer.lat)
    nlons = len(aer.lon)
    nd = np.zeros((nmodes,nsteps,nlevels))           # you will need to make sure that the shape is correct
    rbardry = np.zeros((nmodes,nsteps,nlevels))      # you will need to make sure that the shape is correct
    sigma_g = [1.59,1.59,1.4,2.0,1.59,1.59,2.0]

    # Number densities
    nd[0,:,:] = aer.field34101.values[:,:] # Nuc
    nd[1,:,:] = aer.field34103.values[:,:] # Ait
    nd[2,:,:] = aer.field34107.values[:,:] # Acc
    nd[3,:,:] = aer.field34113.values[:,:] # Coa
    nd[4,:,:] = aer.field34119.values[:,:]  # Ait Insol

    # Dry radius (convert from diameter)
    rbardry[0,:,:] = (aer.field38401.values[:,:] / 2)
    rbardry[1,:,:] = (aer.field38402.values[:,:] / 2)
    rbardry[2,:,:] = (aer.field38403.values[:,:] / 2)
    rbardry[3,:,:] = (aer.field38404.values[:,:] / 2)
    rbardry[4,:,:] = (aer.field38405.values[:,:] / 2)

    # N10
#        # determine if cube, and if so, convert to numpy array
#     if (type(nd_in) == iris.cube.Cube):
#       # extract data to numpy array
#         nd = nd_in.data
#         rbardry = rbardry_in.data
#     else: # input variable is not a cube, assume is already a numpy array
#         nd = nd_in
#         rbardry = rbardry_in

    for imode in range((nd.shape[0])): # loop over number of modes, from size of input nd

        nd_lt_10_this_mode = lognormal_cumulative_to_r(nd[imode],5e-9,rbardry[imode],sigma_g[imode])
        nd_gt_10_this_mode = nd[imode] - nd_lt_10_this_mode
      
        if (imode == 0):
            nd_gt_10 = nd_gt_10_this_mode
        else:
            nd_gt_10 = nd_gt_10 + nd_gt_10_this_mode

   # if input was cube, return cube
   #if (type(nd_in) == iris.cube.Cube)):

   #else: # input variable is not a cube
    nd_gt_10_out = nd_gt_10
    
    N10 = nd_gt_10_out + nd[1:,:,:].sum(axis=0) # add the larger modes
    # add it into the aerosol array 
    N10 = xr.DataArray(N10,coords=[aer.time,aer.z1_hybrid_height],#,aer.lat,aer.lon],
                       dims=['time','z1_hybrid_height'],name='N10')#,'lat','lon'
    aer = aer.assign(N10=N10)
    aer['N10'][:] = N10.values
    
    
    # N3
#        # determine if cube, and if so, convert to numpy array
#     if (type(nd_in) == iris.cube.Cube):
#       # extract data to numpy array
#         nd = nd_in.data
#         rbardry = rbardry_in.data
#     else: # input variable is not a cube, assume is already a numpy array
#         nd = nd_in
#         rbardry = rbardry_in

    for imode in range((nd.shape[0])): # loop over number of modes, from size of input nd

        nd_lt_3_this_mode = lognormal_cumulative_to_r(nd[imode],1.5e-9,rbardry[imode],sigma_g[imode])
        nd_gt_3_this_mode = nd[imode] - nd_lt_3_this_mode
      
        if (imode == 0):
            nd_gt_3 = nd_gt_3_this_mode
        else:
            nd_gt_3 = nd_gt_3 + nd_gt_3_this_mode

   # if input was cube, return cube
   #if (type(nd_in) == iris.cube.Cube)):

   #else: # input variable is not a cube
    nd_gt_3_out = nd_gt_3
    
    N3 = nd_gt_3_out + nd[1:,:,:].sum(axis=0) # add the larger modes
    # add it into the aerosol array 
    N3 = xr.DataArray(N3,coords=[aer.time,aer.z1_hybrid_height],#,aer.lat,aer.lon],
                       dims=['time','z1_hybrid_height'],name='N3')#,'lat','lon'
    aer = aer.assign(N3=N3)
    aer['N3'][:] = N3.values
    
    return aer


# +
# calculates CCN40, CCN50 and CCN60 number concentrations from aerosol model data

def ccn_calcs(aer):
    nsteps = len(aer.time)
    nmodes = 7
    nlevels=len(aer.z1_hybrid_height)
    nlats = len(aer.lat)
    nlons = len(aer.lon)
    nd = np.zeros((nmodes,nsteps,nlevels))#,nlats,nlons))           # you will need to make sure that the shape is correct
    rbardry = np.zeros((nmodes,nsteps,nlevels))#,nlats,nlons))      # you will need to make sure that the shape is correct
    sigma_g = [1.59,1.59,1.4,2.0,1.59,1.59,2.0]

    # Number densities
    nd[0,:,:] = aer.field34101.values[:,:] # Nuc
    nd[1,:,:] = aer.field34103.values[:,:] # Ait
    nd[2,:,:] = aer.field34107.values[:,:] # Acc
    nd[3,:,:] = aer.field34113.values[:,:] # Coa
    nd[4,:,:] = aer.field34119.values[:,:]  # Ait Insol

    # Dry radius (convert from diameter)
    rbardry[0,:,:] = (aer.field38401.values[:,:] / 2)
    rbardry[1,:,:] = (aer.field38402.values[:,:] / 2)
    rbardry[2,:,:] = (aer.field38403.values[:,:] / 2)
    rbardry[3,:,:] = (aer.field38404.values[:,:] / 2)
    rbardry[4,:,:] = (aer.field38405.values[:,:] / 2)
    
    # CCN40
#        # determine if cube, and if so, convert to numpy array
#     if (type(nd_in) == iris.cube.Cube):
#       # extract data to numpy array
#         nd = nd_in.data
#         rbardry = rbardry_in.data
#     else: # input variable is not a cube, assume is already a numpy array
#         nd = nd_in
#         rbardry = rbardry_in

    # CCN40
    for imode in range(4): # only count first four modes, which are the soluble modes in the standard GLOMAP-mode setups

        nd_lt_40_this_mode = lognormal_cumulative_to_r(nd[imode],20e-9,rbardry[imode],sigma_g[imode])
        nd_gt_40_this_mode = nd[imode] - nd_lt_40_this_mode
      
        if (imode == 0):
            nd_gt_40 = nd_gt_40_this_mode
        else:
            nd_gt_40 = nd_gt_40 + nd_gt_40_this_mode

   # if input was cube, return cube
   #if (type(nd_in) == iris.cube.Cube)):

   #else: # input variable is not a cube
    nd_gt_40_out = nd_gt_40
    
    CCN40 = nd_gt_40_out + nd[1:,:,:].sum(axis=0) # add the larger modes
    # add it into the aerosol array 
    CCN40 = xr.DataArray(CCN40,coords=[aer.time,aer.z1_hybrid_height],#,aer.lat,aer.lon],
                       dims=['time','z1_hybrid_height'],name='CCN40')#,'lat','lon'
    aer = aer.assign(CCN40=CCN40)
    aer['CCN40'][:] = CCN40.values
    
   # CCN50
    for imode in range(4): # only count first four modes, which are the soluble modes in the standard GLOMAP-mode setups

        nd_lt_50_this_mode = lognormal_cumulative_to_r(nd[imode],25e-9,rbardry[imode],sigma_g[imode])
        nd_gt_50_this_mode = nd[imode] - nd_lt_50_this_mode
      
        if (imode == 0):
            nd_gt_50 = nd_gt_50_this_mode
        else:
            nd_gt_50 = nd_gt_50 + nd_gt_50_this_mode

   # if input was cube, return cube
   #if (type(nd_in) == iris.cube.Cube)):

   #else: # input variable is not a cube
    nd_gt_50_out = nd_gt_50
    
    CCN50 = nd_gt_50_out + nd[1:,:,:].sum(axis=0) # add the larger modes
    # add it into the aerosol array 
    CCN50 = xr.DataArray(CCN50,coords=[aer.time,aer.z1_hybrid_height],#,aer.lat,aer.lon],
                       dims=['time','z1_hybrid_height'],name='CCN50')#,'lat','lon'
    aer = aer.assign(CCN50=CCN50)
    aer['CCN50'][:] = CCN50.values
    
    # CCN60
    for imode in range(4): # only count first four modes, which are the soluble modes in the standard GLOMAP-mode setups

        nd_lt_60_this_mode = lognormal_cumulative_to_r(nd[imode],30e-9,rbardry[imode],sigma_g[imode])
        nd_gt_60_this_mode = nd[imode] - nd_lt_60_this_mode
      
        if (imode == 0):
            nd_gt_60 = nd_gt_60_this_mode
        else:
            nd_gt_60 = nd_gt_60 + nd_gt_60_this_mode

   # if input was cube, return cube
   #if (type(nd_in) == iris.cube.Cube)):

   #else: # input variable is not a cube
    nd_gt_60_out = nd_gt_60
    
    CCN60 = nd_gt_60_out + nd[1:,:,:].sum(axis=0) # add the larger modes
    # add it into the aerosol array 
    CCN60 = xr.DataArray(CCN60,coords=[aer.time,aer.z1_hybrid_height],#,aer.lat,aer.lon],
                       dims=['time','z1_hybrid_height'],name='CCN60')#,'lat','lon'
    aer = aer.assign(CCN60=CCN60)
    aer['CCN60'][:] = CCN60.values
    
    return aer


# -

def calculate_size_distributions(nmodes,nd,rbardry,t):
    sigma_g = [1.59,1.59,1.4,2.0,1.59,1.59,2.0]

    # determine which modes are active
    mode = np.zeros((nmodes),dtype=bool)
    for imode in range(nmodes):
        mode[imode] = np.isfinite(nd[imode,0]) # if data in array, mode is active
        
    # define points for calculating size distribution
    npts = 50 # number of bins into which interpolate modal output
    rmin = 1.0e-9
    rmax = 1.0e-5
    dryr_mid = np.zeros(npts)
    dryr_int = np.zeros(npts+1)
    for ipt in range (npts+1):
        logr = np.log(rmin)+(np.log(rmax)-np.log(rmin))*np.float(ipt)/np.float(npts)
        dryr_int[ipt] = np.exp(logr)
    for ipt in range (npts):
        dryr_mid[ipt] = 10.0**(0.5*(np.log10(dryr_int[ipt+1])+np.log10(dryr_int[ipt]))) # in m
        
    dndlogd = np.zeros((nmodes+1,npts)) # number of modes, plus total number    
        
    for ipt in range (npts):
        for imode in range (nmodes):

            if (mode[imode]):
                dndlogd[imode,ipt] = lognormal_dndlogd(nd[imode,t],
                                                       dryr_mid[ipt],
                                                       rbardry[imode,t]*2,
                                                       sigma_g[imode])
            else:
                dndlogd[imode,ipt] = 1.0e-15
            
        dndlogd[nmodes,ipt] = np.nansum(dndlogd[0:nmodes,ipt])
        
    return dndlogd,dryr_mid


def lognormal_dndlogd(n,r,rbar,sigma_g):

    # evaluates lognormal distribution dn/dlogd at diameter d
    # dndlogd is the differential wrt the base10 logarithm

    xpi = 3.14159265358979323846e0

    numexp = -(np.log(r)-np.log(rbar))**2.0
    denomexp = 2.0*np.log(sigma_g)*np.log(sigma_g)

    denom = np.sqrt(2.0*xpi)*np.log(sigma_g)

    dndlnd = (n/denom)*np.exp(numexp/denomexp)

    dndlogd = 2.303*dndlnd

    return dndlogd


# +
# calculates aerosol size distributions from aerosol model data

def calc_size_dists(aer):
    nsteps = len(aer.time)
    nmodes = 7
    nd = np.zeros((nmodes,nsteps))
    rbardry = np.zeros ((nmodes,nsteps))

    nd[0,:] = aer.field34101.values[:,0] # Nuc
    nd[1,:] = aer.field34103.values[:,0] # Ait
    nd[2,:] = aer.field34107.values[:,0] # Acc
    nd[3,:] = aer.field34113.values[:,0] # Coa
    nd[4,:] = aer.field34119.values[:,0]  # Ait Insol

    rbardry[0,:] = (aer.field38401.values[:,0] / 2)
    rbardry[1,:] = (aer.field38402.values[:,0] / 2)
    rbardry[2,:] = (aer.field38403.values[:,0] / 2)
    rbardry[3,:] = (aer.field38404.values[:,0] / 2)
    rbardry[4,:] = (aer.field38405.values[:,0] / 2)
    dnd = np.zeros((50,nsteps))

    for i in range(nsteps):

        dndlogd,dryr_mid = calculate_size_distributions(nmodes,nd,rbardry,i)
        dnd[:,i] = dndlogd[7,:]

    dryr_mid = dryr_mid*2.*1.0e9 # m to nm, radius to diameter
    sizedist = xr.DataArray(dnd,coords=[dryr_mid,aer.time],dims=['Dry Diameter','Time']).to_dataset(name='sizedist')
    lat = xr.DataArray(aer.lat,coords=[aer.time],dims=['Time']).to_dataset()
    lon = xr.DataArray(aer.lon,coords=[aer.time],dims=['Time']).to_dataset()

    da = xr.merge([sizedist,lat,lon])
    da['sizedist'] = da['sizedist'].assign_attrs({'Units':'dN / dlogD (cm-3)'})
    da['Dry Diameter'] = da['Dry Diameter'].assign_attrs({'Units':'nm'})
#     da = da.assign_attrs({'history':'Data extracted along {} ship track on {}'.format(
#         shipname,datetime.now().date())})
#     fout = '/g/data/jk72/slf563/ACCESS/output/{}/processed/'.format(job)
#     da.load().to_netcdf(fout+'{}_{}_{}_mean_size_distributions.nc'.format(job,tstep,shipname))
    return da
