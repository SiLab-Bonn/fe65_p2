
import logging

import numpy as np
from bokeh.charts import HeatMap, bins, output_file
from bokeh.palettes import RdYlGn6, RdYlGn9, BuPu9
from bokeh.plotting import figure
import tables as tb
            
def plot_occupancy(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]

        #H, _, _ = np.histogram2d(hit_data['col'], hit_data['row'], bins = (range(65), range(65)))
        #grid = np.indices(H.shape)
        #data = {'column': np.ravel(grid[0]),
        #        'row': np.ravel(grid[1]),
        #        'value': np.ravel(H)
        #       }
        #col = np.ravel(grid[0])
        #row = np.ravel(grid[1])
        #value = np.ravel(H)
        
        hits = hit_data['col'].astype(np.uint16)
        hits = hits * 64
        hits = hits + hit_data['row']
        value = np.bincount(hits)
        value = np.pad(value, (0, 64*64 - value.shape[0]), 'constant')
       
        indices = np.indices(value.shape)
        col = indices[0] / 64
        row = indices[0] % 64
        
        data = {'column': col,
                'row': row,
                'value': value
               }
                
        hm = HeatMap(data, x='column', y='row', values='value', legend='top_right', title='Occupancy', palette=RdYlGn6, stat=None)
        
    return hm, value.reshape((64, 64))

def plot_tot_dist(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]
        
        tot_count = np.bincount(hit_data['tot'])
        tot_count = np.pad(tot_count, (0, 15 - tot_count.shape[0]), 'constant')
        tot_plot = figure(title="ToT Distribution")
        tot_plot.quad(top=tot_count, bottom=0, left=np.arange(-0.5, 14.5, 1), right=np.arange(0.5, 15.5, 1))
    
    return tot_plot, tot_count
    
    
def plot_lv1id_dist(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]
        
        lv1id_count = np.bincount(hit_data['lv1id'])
        lv1id_count = np.pad(lv1id_count, (0, 16 - lv1id_count.shape[0]), 'constant')
        lv1id_plot = figure(title="lv1id Distribution")
        lv1id_plot.quad(top=lv1id_count, bottom=0, left=np.arange(-0.5, 15.5, 1), right=np.arange(0.5, 16.5, 1))
        
    return lv1id_plot, lv1id_count
    
def scan_pix_hist(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        hit_data = in_file_h5.root.hit_data[:]

        param = np.unique(meta_data['scan_param_id'])
        ret = []
        for i in param:
            wh = np.where(hit_data['scan_param_id'] == i)
            hd = hit_data[wh[0]]
            hits = hd['col'].astype(np.uint16)
            hits = hits * 64
            hits = hits + hd['row']
            value = np.bincount(hits)
            value = np.pad(value, (0, 64*64 - value.shape[0]), 'constant')
            #print value
            if len(ret):
                ret = np.vstack((ret, value))
            else:
                ret = value

        s_hist = np.swapaxes(ret,0,1)
        indices = np.indices(s_hist.shape)

        data = {
            'scan_param': np.ravel(indices[1]),
            'pixel': np.ravel(indices[0]),
            'value': np.ravel(s_hist)
        }
        
        hm = HeatMap(data, x='scan_param', y='pixel', values='value', legend='top_right', title='s-scan', palette=BuPu9, stat=None) #, height=4100)
        return hm, s_hist 
    
if __name__ == "__main__":
    pass
