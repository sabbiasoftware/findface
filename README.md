        usage: findface [-h] [-q | -r] [-F FACES_DIR] [-c] [-C COPY_DIR] [-b]
                        [-l LIMIT] [-t TOLERANCE] [-m {0,1}] [-d] -i INCLUDE_FACE
                        [-x EXCLUDE_FACE]
                        SEARCH_DIR

        find images in a directory hierarchy that contain all faces specified

        positional arguments:
          SEARCH_DIR            Directory to search within recursively.

        options:
          -h, --help            show this help message and exit
          -q, --quick           Quick find, skip updating index. When searching in a
                                directory the first time, make sure to use facefind
                                WITHOUT this paramater, to allow building the initial
                                index.
          -r, --rebuild         Force rebuild of indexes from scratch before search.
          -F, --faces-dir FACES_DIR
                                Override default face directory of <dir>/FF_FACES. If
                                relative, then relative to <dir>.
          -c, --copy            Copy images found to a designated destination folder.
                                Default destination directory is <dir>/FF_FACES_FOUND,
                                use -C to override.
          -C, --copy-dir COPY_DIR
                                Override default destination folder of
                                <dir>/FF_FACES_FOUND. If relative, then relative to
                                <dir>.
          -b, --brief           Reduce verbosity and print only path of found images.
          -l, --limit LIMIT     Stop search after finding N images.
          -t, --tolerance TOLERANCE
                                Override default tolerance of 0.6. Lower tolerance
                                value results less faces found.
          -m, --method {0,1}    Choose search method (0 or 1). 0: search for images
                                that have at least one face that has a distance less
                                or equal to tolerance. 1 (default): search for images
                                that have at least one face that has a distance less
                                or equal to tolerance AND no other face in <FACES-DIR>
                                has less distance.
          -d, --debug           Generate ff_trace.html in SEARCH_DIR with face
                                matching details.
          -i, --include-face INCLUDE_FACE
                                Name of face to find. May be specified multiple times.
                                The search returns images that contain ALL faces
                                specified.
          -x, --exclude-face EXCLUDE_FACE
                                Name of face to exclude. May be specified multiple
                                times. Images containing any of these faces are
                                skipped.

            Find images recursively in <dir> that contain ALL specified faces.

            Images in <FACES-DIR> define the list of known faces. A known face with a
            given name can be defined by placing <name>_*.jpg images (1 or more) in
            <FACES-DIR>. Using a handful of face images for a face is supposed to
            increase accuracy. By default <FACES-DIR> is located at <dir>/FF_FACES,
            default location can be overriden by -F.

            Before search images has to be pre-processed (indexed). Initial indexing
            is CPU intense and can take a while for directories with large amount of
            images. Index is stored in <dir>/FF_INDEX.idx. Unless index update is
            disabled by flag -q, the index is always updated before search by adding
            entries for new images and removing entries of missing images.

            During actual search the path of found images is printed preceeded by a
            numeric value showing the distance of found faces. The numeric value can be
            considered as a confidence level, the lower the number indicates the more
            confidence. Use flag -b to produce brief output and print only the path of
            found images.

            Use flag -c to copy found images to a directory <copy-dir>. By default
            <copy-dir> is <dir>/FF_FACES_FOUND, use -C to override. CAUTION: entire
            content of <copy-dir> is purged before each search.

            The face search logic is implemented by the face-recognition module
            (https://pypi.org/project/face-recognition/), a huge thanks to all
            contributors.

            To adjust results:

            - Use -t to change tolerance. If the calculated distance between two faces
              is more than the tolerance, then the faces are considered different.
              Otherwise the faces are considered matching.

            - Use -m to change method. Method 0: search for images that have at least
              one face that has a distance less or equal to tolerance. Method 1: search
              for images that have at least one face that has a distance less or equal
                to tolerance AND no other face in <FACES-DIR> has less distance.

            - For good results make sure to include good quality images in <face-dir>,
              ideally a few images for each face. For further details about the search
              method please consult the documentation of face-recognition module.
