import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy.ndimage import distance_transform_edt as distance
from skimage.draw import ellipse
from skimage import measure


st.set_page_config(page_title='dose master', page_icon="random", layout="wide",
                   initial_sidebar_state="expanded", menu_items=None)



def fall_off(grad,arr,prescrip,mode='linear'):
    
    _arr = distance(arr)

    if mode == 'linear':
        return   grad*_arr
    
    if mode == 'square':
        return grad*np.square(_arr)
    
    #_arr = np.sqrt(_arr)
    
    if mode == 'gaussian':
        return prescrip*(1 - np.exp(-np.square(_arr)/(grad*grad))) #this might not be sensible
    

def dose_boost(crop_targets,oar,dose_grid,grad,max_oar_dose,dose_in,dose_mode):

    dose_out = np.copy(dose_in)

    for target_to_crop,prescription in crop_targets:
        xs,ys = np.nonzero(target_to_crop)

        for ele in zip(xs,ys):
            _arr = np.ones(shape=dose_grid)
            _arr[ele]=0
            _arr = fall_off(grad,_arr, prescription,dose_mode)

            ele_dose = dose_out[ele]-(_arr)

            if max_oar_dose- np.max(ele_dose*oar)>1:
                new_ele_dose = max_oar_dose+np.min(_arr[oar==1])
                dose_out = np.maximum(dose_out, new_ele_dose - _arr)

    return dose_out


def make_structures(dose_grid):
    target = np.zeros(dose_grid, dtype=np.uint8)
    target2 = np.zeros(dose_grid, dtype=np.uint8)
    oar = np.zeros(dose_grid, dtype=np.uint8)

    rr, cc = ellipse(dose_grid[1]//2, dose_grid[1]//2, dose_grid[0]//5, dose_grid[0]//7, rotation=np.deg2rad(0))
    target[rr, cc] = 1

    rr, cc = ellipse(dose_grid[1]//2, dose_grid[1]//2, dose_grid[0]//3, dose_grid[0]//4, rotation=np.deg2rad(0))
    target2[rr, cc] = 1
    target2 = target2*(target==0)


    rr, cc = ellipse(dose_grid[1]//2, dose_grid[1]//2-dose_grid[1]//5, dose_grid[1]//10, dose_grid[1]//10, rotation=np.deg2rad(0))
    oar[rr, cc] = 1


    rr, cc = ellipse(dose_grid[1]//2, dose_grid[1]//2+dose_grid[1]//5, dose_grid[1]//10, dose_grid[1]//10, rotation=np.deg2rad(0))
    oar[rr, cc] = 1

    return target,target2, oar


def calc_dose(dose_grid, target, target2, oar, prescrip, prescrip2, max_oar_dose, dose_mode):

    oar_dist = fall_off(grad,oar==0, prescrip, dose_mode)

    target_to_crop = target*(oar_dist<(prescrip-max_oar_dose))==1
    tech_target = (target_to_crop==0)*target
    tech_dose = prescrip - fall_off(grad,tech_target==0,prescrip,dose_mode)

    target_to_crop2 = target2*(oar_dist<(prescrip2-max_oar_dose))==1
    tech_target2 = (target_to_crop2==0)*target2
    tech_dose2 = prescrip2 - fall_off(grad,tech_target2==0,prescrip,dose_mode)

    tech_dose = np.maximum(tech_dose,tech_dose2)
    tech_dose[tech_dose<0] = 0

    dose = prescrip - fall_off(grad,target==0,prescrip,dose_mode)
    dose2 = prescrip2 - fall_off(grad,target2==0,prescrip,dose_mode)
    dose = np.maximum(dose,dose2)
    dose[dose<0] = 0


    boost_dose = dose_boost([(target_to_crop,prescrip),(target_to_crop2,prescrip)],oar,dose_grid,grad,max_oar_dose,tech_dose,dose_mode)

    return dose, tech_dose, boost_dose, target_to_crop, target_to_crop2




with st.sidebar:

    with st.expander('Dose options'):
        dose_mode = st.pills('Dose falloff model',['linear','square','gaussian'], default='linear')
        grad = st.number_input('dose gradient', value=5.0)

    st.text("")
    st.text("")

    with st.expander('Prescription options'):
        c1,c2 = st.columns(2)
        
        prescrip = c1.number_input('CTV_High presription', value=100)
        prescrip2 = c2.number_input('CTV_Low presription', value=90)
        max_oar_dose = st.number_input('Max OAR dose', value=80)

    st.text("")
    st.text("")

    with st.form('plot_params'):
        st.markdown('Plot options:')

        comp_dose = st.selectbox('select dose to plot',['Full','Basic compromise','Corrected compromise'])

        dose_range = st.slider('colourwash range',value=(0,100), min_value=0, max_value=prescrip)

        st.text("")
        st.markdown("Contour display options")
        show_target = st.checkbox('show ctv_high',value=True)
        show_target2 = st.checkbox('show ctv_low',value=True)
        show_target_crop = st.checkbox('show ctv_high crop')
        show_target_crop2 = st.checkbox('show ctv_low crop ')
        show_oar = st.checkbox('show oar',value=True)

        st.form_submit_button('Update plots')


dose_grid = (100,100)




target,target2, oar = make_structures(dose_grid)


dose, tech_dose, boost_dose, target_to_crop, target_to_crop2 = calc_dose(dose_grid, target, target2, oar, prescrip, prescrip2, max_oar_dose, dose_mode)

#make contours - only used to display on plot
target_contour = measure.find_contours(target, 0.5)
oar_contour = measure.find_contours(oar, 0.5)
target_to_crop_contour = measure.find_contours(target_to_crop, 0.5)

target2_contour = measure.find_contours(target2, 0.5)
oar_contour = measure.find_contours(oar, 0.5)
target_to_crop2_contour = measure.find_contours(target_to_crop2, 0.5)



st.markdown(f'### {comp_dose}')

col1,col2 = st.columns((1.2,1),vertical_alignment='center')

fig, ax = plt.subplots()

dose_dict = {'Full':dose, 'Basic compromise':tech_dose ,'Corrected compromise':boost_dose}


plot_dose = dose_dict[comp_dose]



if show_oar:
    for cont in oar_contour:
        plt.plot([x[1] for x in cont],[x[0]  for x in cont],color='red',label='oar')

if show_target_crop:
    for cont in target_to_crop_contour:
        plt.plot([x[1] for x in cont],[x[0]  for x in cont],color='yellow',label='ctv_high crop')

if show_target2:
    for cont in target2_contour:
        plt.plot([x[1] for x in cont],[x[0]  for x in cont],color='cyan',label='ctv_low')

if show_target_crop2:
    for cont in target_to_crop2_contour:
        plt.plot([x[1] for x in cont],[x[0]  for x in cont],color='orange',label='ctv_low crop')

if show_target:
    for cont in target_contour:
        plt.plot([x[1] for x in cont],[x[0]  for x in cont],color='green',label='ctv_high')


pl = ax.imshow(plot_dose,origin='lower',vmin=dose_range[0],vmax=dose_range[1])
plt.colorbar(pl,ax=ax)
plt.axis('off')

plt.legend()

handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax.legend(*zip(*unique))

col1.pyplot(fig)



fig, ax = plt.subplots()

if show_oar:
    plt.hist(plot_dose[oar==1],bins=1000,density=True,cumulative=-1,histtype='step',
            color='red', label='oar',range=(0,prescrip))

if show_target_crop:
    plt.hist(plot_dose[target_to_crop==1],bins=1000,density=True,cumulative=-1,histtype='step',
            color='yellow', label='ctv_high crop',range=(0,prescrip))

if show_target_crop2:
    plt.hist(plot_dose[target_to_crop2==1],bins=1000,density=True,cumulative=-1,histtype='step',
            color='orange', label='ctv_low crop',range=(0,prescrip))

if show_target2:
    plt.hist(plot_dose[target2==1],bins=1000,density=True,cumulative=-1,histtype='step',
         color='cyan', label='ctv_low',range=(0,prescrip))

if show_target:
    plt.hist(plot_dose[target==1],bins=1000,density=True,cumulative=-1,histtype='step',
         color='green', label='ctv_high',range=(0,prescrip))





ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.xlabel("Dose")
plt.ylabel("Volume")
plt.legend()


col2.pyplot(fig)