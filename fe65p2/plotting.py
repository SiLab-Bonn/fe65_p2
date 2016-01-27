
import logging

import numpy as np
from bokeh.charts import HeatMap, bins, output_file, vplot, hplot
from bokeh.palettes import RdYlGn6, RdYlGn9, BuPu9
from bokeh.plotting import figure
import tables as tb
import analysis as analysis
import yaml

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
                
        hm = HeatMap(data, x='column', y='row', values='value', legend='top_right', title='Occupancy', palette=RdYlGn6[::-1], stat=None)
        
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
        
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        scan_range = scan_args['scan_range']
        scan_range_inx = np.arange(scan_range[0], scan_range[1], scan_range[2])
        
        repeat_command = scan_args['repeat_command']
        
        np.set_printoptions(threshold=np.nan)
        k = 5
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
            if len(ret):
                ret = np.vstack((ret, value))
            else:
                ret = value

        s_hist = np.swapaxes(ret,0,1)
        indices = np.indices(s_hist.shape)

        param_inx = np.ravel(indices[1].astype(np.float64))#*0.05 - 0.6)
        param_inx_string = param_inx.astype('|S5') 

        #data = {
        #    'scan_param': param_inx_string, 
        #    'pixel': np.ravel(indices[0]),
        #    'value': np.ravel(s_hist)
        #}
        #hm = HeatMap(data, x='scan_param', y='pixel', values='value', legend='top_right', title='s-scan', palette=BuPu9, stat=None) #, height=4100)
        
        
        pix_scan_hist = np.empty((s_hist.shape[1],repeat_command + 20))
        for param in range(s_hist.shape[1]):
            h_count = np.bincount(s_hist[:,param])
            h_count = h_count[:repeat_command]
            pix_scan_hist[param] = np.pad(h_count, (0, (repeat_command + 20) - h_count.shape[0]), 'constant')
        data = {
            'scan_param': np.ravel(np.indices(pix_scan_hist.shape)[0]), 
            'count': np.ravel(np.indices(pix_scan_hist.shape)[1]),
            'value': np.ravel(pix_scan_hist)
        }
        hm1 = HeatMap(data, x='scan_param', y='count', values='value', legend='bottom_right', title='scan_histo', palette=BuPu9, stat=None) #, height=4100)
         
        mean = np.empty(64*64)
        noise = np.empty(64*64)
        x = scan_range_inx 
        for pix in range (64*64):
            mu, sigma = analysis.fit_scurve(s_hist[pix], x)
            mean[pix] = mu
            noise[pix] = sigma

        px = 457
        single_scan = figure(title="Single pixel scan " + str(px) )
        single_scan.diamond(x=x, y=s_hist[px], size=5, color="#1C9099", line_width=2)
        yf = analysis.scurve(x, 100, mean[px], noise[px])
        single_scan.cross(x=x, y=yf, size=5, color="#E6550D", line_width=2)    
        
        mean[mean > scan_range_inx[-1]] = 0
        hm2 = figure(title="Threshold [V]", plot_width=1000)
        hm2.diamond(y=mean, x=range(64*64), size=1, color="#1C9099", line_width=2)
        hist, edges = np.histogram(mean, density=True, bins=50)
        p1 = figure(title="Threshold Distribution [V]")
        p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#036564", line_color="#033649",)
        
        
        noise[noise > 0.02] = 0 #this should be done based on 6sigma?
        hm3 = figure(title="Noise [V]", plot_width=1000)
        hm3.diamond(y=noise, x=range(64*64), size=2, color="#1C9099", line_width=2)
        hist, edges = np.histogram(noise, density=True, bins=50)
        p2 = figure(title="Noise Distribution [V]")
        p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#036564", line_color="#033649",)
        
        return vplot(hplot(hm2, p1), hplot(hm3,p2), hplot(hm1, single_scan) ), s_hist 
    
if __name__ == "__main__":
    pass
