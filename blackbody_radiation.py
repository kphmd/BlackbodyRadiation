import taichi as ti

ti.init(arch=ti.cpu)

# ref1: https://www.fourmilab.ch/documents/specrend/
# ref2: https://www.fourmilab.ch/documents/specrend/specrend.c

res_x = 512
res_y = 512
pixels = ti.Vector.field(3, ti.f32, shape=(res_x,res_y))
cie_xyz = ti.Vector.field(3, ti.f32, shape=(res_x,res_y))

Temperature = ti.field(ti.f32, shape=())
Temperature[None] = 6500
RGB = ti.Vector.field(3, ti.f32, shape=())

xR,yR, xG,yG, xB,yB, xW,yW, Gamma= 0.630, 0.340,  0.310, 0.595,  0.155, 0.070,  0.3127, 0.3291,  0.0 # SMPTEsystem
cie_colour_match = ti.Vector.field(n=3, dtype=ti.f32, shape=(81))

@ti.func 
def frac(x) :
    return x - ti.floor(x)

@ti.func
def bb_spectrum(wavelength,bbTemp) :
    '''
    wavelength (nm)
    bbTemp (K)
    '''
    wlm = wavelength * 1e-9 # Wavelength in meters
    return (3.74183e-16 * ti.pow(wlm, -5.0)
    ) / (ti.exp(1.4388e-2 / (wlm * bbTemp)) - 1.0)

@ti.kernel
def init_cie_colour_match() :
    x = ti.static(cie_colour_match)
    i =0; x[i] = [0.0014,0.0000,0.0065]; i+=1; x[i] = [0.0022,0.0001,0.0105]
    i+=1; x[i] = [0.0042,0.0001,0.0201]; i+=1; x[i] = [0.0076,0.0002,0.0362]
    i+=1; x[i] = [0.0143,0.0004,0.0679]; i+=1; x[i] = [0.0232,0.0006,0.1102]
    i+=1; x[i] = [0.0435,0.0012,0.2074]; i+=1; x[i] = [0.0776,0.0022,0.3713]
    i+=1; x[i] = [0.1344,0.0040,0.6456]; i+=1; x[i] = [0.2148,0.0073,1.0391]
    i+=1; x[i] = [0.2839,0.0116,1.3856]; i+=1; x[i] = [0.3285,0.0168,1.6230]
    i+=1; x[i] = [0.3483,0.0230,1.7471]; i+=1; x[i] = [0.3481,0.0298,1.7826]
    i+=1; x[i] = [0.3362,0.0380,1.7721]; i+=1; x[i] = [0.3187,0.0480,1.7441]
    i+=1; x[i] = [0.2908,0.0600,1.6692]; i+=1; x[i] = [0.2511,0.0739,1.5281]
    i+=1; x[i] = [0.1954,0.0910,1.2876]; i+=1; x[i] = [0.1421,0.1126,1.0419]
    i+=1; x[i] = [0.0956,0.1390,0.8130]; i+=1; x[i] = [0.0580,0.1693,0.6162]
    i+=1; x[i] = [0.0320,0.2080,0.4652]; i+=1; x[i] = [0.0147,0.2586,0.3533]
    i+=1; x[i] = [0.0049,0.3230,0.2720]; i+=1; x[i] = [0.0024,0.4073,0.2123]
    i+=1; x[i] = [0.0093,0.5030,0.1582]; i+=1; x[i] = [0.0291,0.6082,0.1117]
    i+=1; x[i] = [0.0633,0.7100,0.0782]; i+=1; x[i] = [0.1096,0.7932,0.0573]
    i+=1; x[i] = [0.1655,0.8620,0.0422]; i+=1; x[i] = [0.2257,0.9149,0.0298]
    i+=1; x[i] = [0.2904,0.9540,0.0203]; i+=1; x[i] = [0.3597,0.9803,0.0134]
    i+=1; x[i] = [0.4334,0.9950,0.0087]; i+=1; x[i] = [0.5121,1.0000,0.0057]
    i+=1; x[i] = [0.5945,0.9950,0.0039]; i+=1; x[i] = [0.6784,0.9786,0.0027]
    i+=1; x[i] = [0.7621,0.9520,0.0021]; i+=1; x[i] = [0.8425,0.9154,0.0018]
    i+=1; x[i] = [0.9163,0.8700,0.0017]; i+=1; x[i] = [0.9786,0.8163,0.0014]
    i+=1; x[i] = [1.0263,0.7570,0.0011]; i+=1; x[i] = [1.0567,0.6949,0.0010]
    i+=1; x[i] = [1.0622,0.6310,0.0008]; i+=1; x[i] = [1.0456,0.5668,0.0006]
    i+=1; x[i] = [1.0026,0.5030,0.0003]; i+=1; x[i] = [0.9384,0.4412,0.0002]
    i+=1; x[i] = [0.8544,0.3810,0.0002]; i+=1; x[i] = [0.7514,0.3210,0.0001]
    i+=1; x[i] = [0.6424,0.2650,0.0000]; i+=1; x[i] = [0.5419,0.2170,0.0000]
    i+=1; x[i] = [0.4479,0.1750,0.0000]; i+=1; x[i] = [0.3608,0.1382,0.0000]
    i+=1; x[i] = [0.2835,0.1070,0.0000]; i+=1; x[i] = [0.2187,0.0816,0.0000]
    i+=1; x[i] = [0.1649,0.0610,0.0000]; i+=1; x[i] = [0.1212,0.0446,0.0000]
    i+=1; x[i] = [0.0874,0.0320,0.0000]; i+=1; x[i] = [0.0636,0.0232,0.0000]
    i+=1; x[i] = [0.0468,0.0170,0.0000]; i+=1; x[i] = [0.0329,0.0119,0.0000]
    i+=1; x[i] = [0.0227,0.0082,0.0000]; i+=1; x[i] = [0.0158,0.0057,0.0000]
    i+=1; x[i] = [0.0114,0.0041,0.0000]; i+=1; x[i] = [0.0081,0.0029,0.0000]
    i+=1; x[i] = [0.0058,0.0021,0.0000]; i+=1; x[i] = [0.0041,0.0015,0.0000]
    i+=1; x[i] = [0.0029,0.0010,0.0000]; i+=1; x[i] = [0.0020,0.0007,0.0000]
    i+=1; x[i] = [0.0014,0.0005,0.0000]; i+=1; x[i] = [0.0010,0.0004,0.0000]
    i+=1; x[i] = [0.0007,0.0002,0.0000]; i+=1; x[i] = [0.0005,0.0002,0.0000]
    i+=1; x[i] = [0.0003,0.0001,0.0000]; i+=1; x[i] = [0.0002,0.0001,0.0000]
    i+=1; x[i] = [0.0002,0.0001,0.0000]; i+=1; x[i] = [0.0001,0.0000,0.0000]
    i+=1; x[i] = [0.0001,0.0000,0.0000]; i+=1; x[i] = [0.0001,0.0000,0.0000]
    i+=1; x[i] = [0.0000,0.0000,0.0000]; i+=1; x[i] = [0.0000,0.0000,0.0000]
    
@ti.func
def wavelength_to_xyz(wavelength, bbTemp) :
    Me = bb_spectrum(wavelength,bbTemp)
    i = int(wavelength-380) // 5
    return Me * cie_colour_match[i]

@ti.func
def ray_shoot(wavelength, bbTemp, sx, sy) :
    xyz = wavelength_to_xyz(wavelength,bbTemp)
    PI = ti.acos(-1.0)
    t = (wavelength - 380) / (780 - 380)
    # N1 = 1.5 * (t) + 1.54 * (1-t) 
    # N1 = 1.3 * (t) + 1.7 * (1-t) 
    N1 = 1.3 * (t) + 2.7 * (1-t) 

    deg_in = 60*PI/180
    deg_out = ti.asin(ti.sin(deg_in)/N1)

    dx = ti.cos(PI/2-deg_in)
    dy = ti.sin(PI/2-deg_in)

    dxo = ti.cos(PI/2-deg_out)
    dyo = ti.sin(PI/2-deg_out)

    x, y = 0.0 + sx, 0.0 + sy
    while x < res_x and y < res_y :
        i, j = int(x), int(res_y-1-y)
        cie_xyz[i,j] += xyz

        if y > res_y // 2:
            dx,dy = dxo,dyo
            
        x += dx
        y += dy

@ti.func
def xyz_to_rgb(xyz) :
    xyzMat = ti.Matrix([
        [       xR,       xG,       xB],
        [       yR,       yG,       yB],
        [1-(xR+yR),1-(xG+yG),1-(xB+yB)]
    ])

    xyzW = ti.Vector([xW,yW,1-(xW+yW)])
    rgbMat = xyzMat.inverse() 

    rgbW = (rgbMat @ xyzW) / yW
    rgbW /= ti.max(rgbW.x,ti.max(rgbW.y,rgbW.z))

    rgbMat = rgbMat / ti.Matrix.cols([rgbW,rgbW,rgbW])
    rgb = rgbMat @ xyz 

    return rgb


@ti.func
def gamma_correct(c) :
    cc = 0.018
    if c < cc :
        c *= ((1.099 * pow(cc, 0.45)) - 0.099) / cc
    else :
        c = (1.099 * ti.pow(c, 0.45)) - 0.099
    return c

@ti.func 
def gamma_correct_rgb(rgb) :
    rgb.x = gamma_correct(rgb.x)
    rgb.y = gamma_correct(rgb.y)
    rgb.z = gamma_correct(rgb.z)
    return rgb

@ti.func
def constrain_rgb(rgb) :
    w = ti.min(rgb.x,ti.min(rgb.y,rgb.z))
    if w < 0 :
        rgb -= w
    return rgb 

@ti.func
def norm_rgb(rgb) :
    w = ti.max(rgb.x,ti.max(rgb.y,rgb.z))
    if w > 0 :
        rgb /= w
    return rgb 

@ti.func
def render_rgb(xyz) :
    rgb = xyz_to_rgb(xyz)
    rgb = constrain_rgb(rgb)
    rgb = norm_rgb(rgb)
    rgb = gamma_correct_rgb(rgb)
    return rgb

@ti.func 
def update_rays(bbTemp) :
    for i,j in cie_xyz :
        cie_xyz[i,j] *= 0

    sx,sy = 0, res_y//4
    
    for i in range((780-380)//5+1) :
        wl = 380 + i * 5 
        ray_shoot(wl, bbTemp, sx, sy)
        for j in range(1,3) :
            # ray_shoot(wl, bbTemp, 0, j)
            ray_shoot(wl, bbTemp, sx+j, sy)

    xyz_max = 0.0
    for i,j in cie_xyz :
        xyz = cie_xyz[i,j]
        xyz_max = ti.max(xyz_max,(xyz.x + xyz.y + xyz.z))

    for i,j in cie_xyz :
        xyz = cie_xyz[i,j]
        xyz_base = (xyz.x + xyz.y + xyz.z)
        pixels[i,j] = render_rgb(xyz/xyz_max) * ti.pow(xyz_base / xyz_max, 0.2)
    
    return pixels[sx, res_y-1-sy]

@ti.kernel
def render(t:ti.f32) :
    RGB[None] = update_rays(Temperature[None])
    # print(rgb)

    for i_,j_ in pixels:
        if res_y - 1 - j_ < 64 :
            pixels[i_,j_] = RGB[None] # ti.Vector([1.0,1.0,1.0])


gui = ti.GUI("Blackbody Radiation", res=(res_x,res_y))
gui.fps_limit = 24

init_cie_colour_match()
t = 0.0
while True :
    t += 1e-3
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'h':
            Temperature[None] += 500
        elif e.key == 'c' :
            if Temperature[None] <= 2000 :
                continue
            Temperature[None] -= 500

    render(t)
    gui.set_image(pixels)

    gui.text(content=f'Temperature = {Temperature[None]} K (press c: cold,press h: hot)', pos=(0.05, 0.99), color=0x7f7f7f)
    gui.text(content=f'RGB = {RGB[None]}', pos=(0.05, 0.95), color=0x7f7f7f)

    gui.show()
