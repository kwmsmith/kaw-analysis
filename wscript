top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_c')
    opt.load('python')
    opt.load('cython')

def configure(conf):
    conf.load('compiler_c')
    conf.load('python')
    conf.check_python_headers()
    conf.check_python_module('numpy')
    conf.check_numpy_version(minver=(1,3))
    conf.get_numpy_includes()
    conf.load('cython')

def build(bld):
    # first try to build a C-based cython extension
    # bld(
        # features = 'c cshlib pyext',
        # source   = 'wrap_gsl_interp.pyx',
        # target   = 'wrap_gsl_interp',
        # includes = '/opt/local/include',
        # lib      = 'gsl gslcblas',
        # libpath = '/opt/local/lib',
        # use      = "NUMPY",
        # )

    bld(
        features = "c cshlib pyext",
        source = "rad_prof.c _rad_prof.pyx",
        target = "_rad_prof",
        includes = "..",
        use = "NUMPY",
        )


from waflib.Configure import conf
@conf
def check_numpy_version(conf, minver, maxver=None):
    conf.start_msg("Checking numpy version")
    minver = tuple(minver)
    if maxver: maxver = tuple(maxver)
    (np_ver_str,) = conf.get_python_variables(conf.env['PYTHON'],
            ['numpy.version.short_version'], ['import numpy'])
    np_ver = tuple([int(x) for x in np_ver_str.split('.')])
    if np_ver < minver or (maxver and np_ver > maxver):
        conf.end_msg(False)
        conf.fatal("numpy version %s is not in the "
                "range of supported versions: minimum=%s, maximum=%s" % (np_ver_str, minver, maxver))
    conf.end_msg(str(np_ver))

@conf
def get_numpy_includes(conf):
    conf.start_msg("Checking numpy includes")
    (np_includes,) = conf.get_python_variables(conf.env['PYTHON'],
            ['numpy.get_include()'], ['import numpy'])
    conf.env.INCLUDES_NUMPY = np_includes
    conf.end_msg('ok (%s)' % np_includes)

# vim:ft=python
