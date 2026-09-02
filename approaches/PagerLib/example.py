from pagerlib.file_input import FileInput
from pagerlib.extractors.page_extractor import PDFIMGExtractor, Rows2Regions
from pagerlib.dtypes import ImageSegment, Image

fi = FileInput()
pdf_extr_img = PDFIMGExtractor()
rows2regions = Rows2Regions()


prdf = fi('/home/dataset/publaynet_dev/publaynet_pdfs/pdfs/dev/PMC4943357_00002.pdf')
pdf_extr_img.extract(prdf)
rows2regions.extract(prdf)

page0 = prdf.data['pages'][0]
print('size:', page0.segment.width, page0.segment.height)
'''size: 596 794'''
segments:list[ImageSegment] =  [reg.segment for reg in page0.children[1:]]

def get_label(reg):
    if 'label' in reg.data:
        return reg.data['label']
    if isinstance(reg, Image):
        return 'figure'
    return 'other'

labels: list[str] = [get_label(reg) for reg in page0.children[1:]]
for lab, seg in zip(labels, segments):
    print(lab, seg.get_segment_p_size())
'''Вернет
text {'x_top_left': 42, 'y_top_left': 55, 'height': 36, 'width': 250}
text {'x_top_left': 42, 'y_top_left': 93, 'height': 152, 'width': 249}
text {'x_top_left': 42, 'y_top_left': 247, 'height': 75, 'width': 249}
text {'x_top_left': 42, 'y_top_left': 324, 'height': 151, 'width': 249}
text {'x_top_left': 42, 'y_top_left': 478, 'height': 100, 'width': 249}
header {'x_top_left': 42, 'y_top_left': 602, 'height': 11, 'width': 41}
text {'x_top_left': 42, 'y_top_left': 619, 'height': 113, 'width': 249}
other {'x_top_left': 542, 'y_top_left': 19, 'height': 7, 'width': 10}
text {'x_top_left': 306, 'y_top_left': 55, 'height': 114, 'width': 249}
text {'x_top_left': 306, 'y_top_left': 172, 'height': 23, 'width': 249}
table {'x_top_left': 306, 'y_top_left': 215, 'height': 513, 'width': 223}
'''