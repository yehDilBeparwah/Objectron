def gen_gaussian_heatmap(item, num_kps, input_shape = [],output_shape=[], sigma=1.5):
    '''
    item: incoming sample :'dict'
    num_kps: number of key points :'int'
    input_shape: 2D tuple/list describing input dimensions :'tuple/list'
    output_shape: 2D tuple/list describing expected output dimensions :'tuple/list'
    sigma: standard deviation for 2D gaussian
    '''
    #xc,yc = [300.5,400.4,535.6], [80.2, 67.8, 89.9] List of key points co-ordinates(size 29 for FLIC dataset)
    xc,yc = item['xcoords'], item['ycoords']
    x = np.arange(0.0,output_shape[0])
    y = np.arange(0.0,output_shape[1])
    # tf.meshgrid([0,1,2],[0,1,2]) creates a 3x3 2d grid with 1 step increments starting 
    # from the lower bound(0 in this example)  
    # xx,yy = [[0,1,2],[0,1,2],[0,1,2]], [[0,0,0],[1,1,1],[2,2,2]]
    xx,yy = tf.meshgrid(x,y)
    # xx.shape = (10,10,1) to aid with broadcasting as xc is reshaped to (1,1,29).
    # Resultant 10 x 10 grid is replicated 29 times to produce gaussians centered at key points
    xx = tf.reshape(xx, (*output_shape,1))
    yy = tf.reshape(yy, (*output_shape,1))
    # keypoint are noted in original input dimensions, need to map to output dimensions       
    x_mean = tf.floor(tf.reshape(xc[:],[1,1,num_kps]) / input_shape[0] * output_shape[0] + 0.5)
    y_mean = tf.floor(tf.reshape(yc[:],[1,1,num_kps]) / input_shape[1] * output_shape[1] + 0.5)
    
    heatmap = tf.exp(-(((xx-x_mean)/sigma)**2)/2 -(((yy-y_mean)/sigma)**2)/2)
    
    return {'image':item['image'],
            'target':tf.cast(heatmap, tf.float32),
            'valid_mask':tf.cast(tf.math.is_nan(x_mean),tf.float32)}
