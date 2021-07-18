from multispectral import Frame, Unmixing

def executePCA(msiName, msiPath, outputPath, n_components):
    # collect images in root_dir matching regex; groups 1 and 2 of the match object
    # identify the document and the layer respectively (optional)
    frame = Frame(root_dir=msiPath,
                  regex='(' + msiName + ')_(\d+).png',
                  group_framename=1,
                  group_layername=2)

    # make unmixing object: loads images of frame and converts them to a data matrix
    um = Unmixing(frame)
    # perform principal component analysis, store visualizations of first 5 components
    # in given output folder (or by default frame.root_dir/pca), return frame containing those
    principal_components = um.unmix(method=Unmixing.Method.PCA, n_components=n_components, out_dir=outputPath,
                                    out_extension='png',
                                    verbose=True)
