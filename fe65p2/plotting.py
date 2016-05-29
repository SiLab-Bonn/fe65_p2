
import logging

import numpy as np
from bokeh.charts import HeatMap, bins, output_file, vplot, hplot
from bokeh.palettes import RdYlGn6, RdYlGn9, BuPu9, Spectral11
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

import tables as tb
import analysis as analysis
import yaml
from progressbar import ProgressBar

def plot_status(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        kwargs = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
        power_status = yaml.load(in_file_h5.root.meta_data.attrs.power_status)

    data = { 'nx': [], 'value': [] };
    
    data['nx'].append('Scan Parameters:')
    data['value'].append('')
        
    for key, value in kwargs.iteritems():
        data['nx'].append(key)
        data['value'].append(value)
    
    data['nx'].append('DACs settings:')
    data['value'].append('')

    for key, value in dac_status.iteritems():
        data['nx'].append(key)
        data['value'].append(value)
    
    data['nx'].append('Power Status:')
    data['value'].append('')
    
    for key, value in power_status.iteritems():
        data['nx'].append(key)
        data['value'].append("{:.2f}".format(value))
        
    source = ColumnDataSource(data)

    columns = [
            TableColumn(field="nx", title="Name"),
            TableColumn(field="value", title="Value"),
        ]
    
    data_table = DataTable(source=source, columns=columns, width=300)

    return data_table
            
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
    

def scan_pix_hist(h5_file_name, scurve_sel_pix = 200):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        hit_data = in_file_h5.root.hit_data[:]
        en_mask = in_file_h5.root.scan_masks.en_mask[:]
        
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        scan_range = scan_args['scan_range']
        scan_range_inx = np.arange(scan_range[0], scan_range[1], scan_range[2])
        
        repeat_command = scan_args['repeat_command']
        
        np.set_printoptions(threshold=np.nan)
        k = 5
        param = np.unique(meta_data['scan_param_id'])
        ret = []
        for i in param:
            wh = np.where(hit_data['scan_param_id'] == i) #this can be faster and multi threaded
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

        en_mask = np.ravel(en_mask)
        
        #data = {
        #    'scan_param': param_inx_string, 
        #    'pixel': np.ravel(indices[0]),
        #    'value': np.ravel(s_hist)
        #}
        #hm = HeatMap(data, x='scan_param', y='pixel', values='value', legend='top_right', title='s-scan', palette=BuPu9, stat=None) #, height=4100)
        
        
        pix_scan_hist = np.empty((s_hist.shape[1],repeat_command + 10))
        for param in range(s_hist.shape[1]):
            h_count = np.bincount(s_hist[:,param])
            h_count = h_count[:repeat_command+10]
            pix_scan_hist[param] = np.pad(h_count, (0, (repeat_command + 10) - h_count.shape[0]), 'constant')
        
        log_hist = np.log10(pix_scan_hist)
        log_hist[~np.isfinite(log_hist)] = 0
        data = {
            'scan_param': np.ravel(np.indices(pix_scan_hist.shape)[0]), 
            'count': np.ravel(np.indices(pix_scan_hist.shape)[1]),
            'value': np.ravel(log_hist)
        }
        
        hm1 = HeatMap(data, x='scan_param', y='count', values='value', title='s-scans', palette=Spectral11[::-1], stat=None, plot_width=1000) #, height=4100)
         
        mean = np.empty(64*64)
        noise = np.empty(64*64)
        x = scan_range_inx
        for pix in range (64*64):
            mu, sigma = analysis.fit_scurve(s_hist[pix], x) #this can multi threaded
            mean[pix] = mu
            noise[pix] = sigma
       
        px = scurve_sel_pix #1110 #1539
        single_scan = figure(title="Single pixel scan " + str(px) )
        single_scan.diamond(x=x, y=s_hist[px], size=5, color="#1C9099", line_width=2)
        yf = analysis.scurve(x, 100, mean[px], noise[px])
        single_scan.cross(x=x, y=yf, size=5, color="#E6550D", line_width=2)    
        
        hm_th = figure(title="Threshold", x_axis_label = "pixel #", y_axis_label = "threshold [V]", y_range=(scan_range_inx[0], scan_range_inx[-1]), plot_width=1000)
        hm_th.diamond(y=mean, x=range(64*64), size=1, color="#1C9099", line_width=2)
        hm_th.extra_y_ranges = {"e": Range1d(start=scan_range_inx[0]*1000*8.0, end=scan_range_inx[-1]*1000*8.0)}
        hm_th.add_layout(LinearAxis(y_range_name="e"), 'right')

        mean = mean[en_mask]
        
        mean[mean > scan_range_inx[-1]] = 0
        hist, edges = np.histogram(mean, density=True, bins=50)
        
        plt_th_dist = figure(title="Threshold Distribution", x_axis_label = "threshold [V]", y_range=(0, 1.1*np.max(hist[1:])))
        plt_th_dist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#036564", line_color="#033649", legend="# {0:d}  mean={1:.2f}  std={2:.2f}".format(len(mean), np.mean(mean)*8*1000, np.std(mean)*8*1000))
        plt_th_dist.extra_x_ranges = {"e": Range1d(start=edges[0]*1000*8.0, end=edges[-1]*1000*8.0)}
        plt_th_dist.add_layout(LinearAxis(x_range_name="e"), 'above')
        plt_th_dist.legend.glyph_width = 0
        

        
        noise_hist = noise[en_mask]
        
        mean_noise = np.mean(noise_hist)
        std_noise = np.std(noise_hist)
        noise_hist = noise_hist[noise_hist < mean_noise + std_noise*6] #cut at 5 sigma
        
        noise[noise > 0.02] = 0.02 #this should be done based on 6sigma?
                
        hm_noise = figure(title="Noise", x_axis_label = "pixel #", y_axis_label = "noise [V]", y_range=(0, np.max(noise)), plot_width=1000)
        hm_noise.diamond(y=noise, x=range(64*64), size=2, color="#1C9099", line_width=2)
        hm_noise.extra_y_ranges = {"e": Range1d(start=0, end=np.max(noise)*1000*8.0)}
        hm_noise.add_layout(LinearAxis(y_range_name="e"), 'right')
        
    
        hist, edges = np.histogram(noise_hist, density=True, bins=50)

        plt_noise_dist = figure(title="Noise Distribution", x_axis_label = "noise [V]", y_range=(0, 1.1*np.max(hist[1:])))
        plt_noise_dist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#036564", line_color="#033649", legend="# {0:d}  mean={1:.2f}  std={2:.2f}".format(len(noise_hist), np.mean(noise_hist)*8*1000, np.std(noise_hist)*8*1000))
        plt_noise_dist.extra_x_ranges = {"e": Range1d(start=edges[0]*1000*8.0, end=edges[-1]*1000*8.0)}
        plt_noise_dist.add_layout(LinearAxis(x_range_name="e"), 'above')
        plt_noise_dist.legend.glyph_width = 0
        
        return vplot(hplot(hm_th, plt_th_dist), hplot(hm_noise,plt_noise_dist), hplot(hm1, single_scan) ), s_hist 
    
if __name__ == "__main__":
    pass
