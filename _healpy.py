import math
import healpy.pixelfunc

def latlon2healpix( lat , lon , res ):
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0
    xs = ( math.cos(lat) * math.cos(lon) )
    ys = ( math.cos(lat) * math.sin(lon) )
    zs = ( math.sin(lat) )
    return healpy.pixelfunc.vec2pix( int(res) , xs , ys , zs )

def healpix2latlon( code , res ):
    [xs, ys, zs] = healpy.pix2vec( int(res) , code )
    lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
    lon = float( math.atan2(ys, xs) * 180.0 / math.pi )
    return [ lat , lon ]
