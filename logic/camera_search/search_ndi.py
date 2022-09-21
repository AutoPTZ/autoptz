import NDIlib as ndi


def get_ndi_sources():
    if not ndi.initialize():
        return 0

    ndi_find = ndi.find_create_v2()

    if ndi_find is None:
        return 0
    ndi.find_wait_for_sources(ndi_find, 5000)
    sources = []

    if not ndi.find_wait_for_sources(ndi_find, 5000):
        print('No change to the sources found.')

    sources = ndi.find_get_current_sources(ndi_find)
    print('Network sources (%s found).' % len(sources))
    for i, s in enumerate(sources):
        print('%s. %s' % (i + 1, s.ndi_name))

    return sources