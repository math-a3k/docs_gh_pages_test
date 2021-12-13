import os,  numpy as np
from typing import Dict, Mapping




def test_convert_to_yolov5():
    info_dict = { 'bboxes': [{'class': 'AIRPLANE', 'xmin': 171, 'ymin': 161, 'xmax': 185, 'ymax': 172}], 'filename': 'IR_AIRPLANE_0511_297.png', 'image_size': (320, 256, 1) }
    names = {"AIRPLANE":0}
    convert_to_yolov5(info_dict,names, output= "./testdata/util_yolo")


def test_yolov5_from_xml():
    yolo_extract_info_from_xml("testdata/util_yolo/724_29873.xml")
    yolov5_from_xml(xml_file_path= "testdata/util_yolo/724_29873.xml",output= "testdata/util_yolo")
    yolov5_from_xml(xml_folder="testdata/util_yolo",output= "testdata/util_yolo")


def test_all():
    # test_extract_info_from_xml()
    test_convert_to_yolov5()
    test_yolov5_from_xml()

    
##############################################################################################################
# Function to get the data from XML Annotation
def yolo_extract_info_from_xml( xml_file:str)->Mapping:
    """
        extracted data from Xml file formated in VOC

        parameter:
        xml_file(str): xml file path which need to transform

        returns: 
        dict: returns all the extracted information from xml which is important in transforming to any format.

    """
    import xml.etree.ElementTree as ET

    names = {}
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    if subelem.text not in names.keys():
                      names[subelem.text]=len(names.keys())
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)

    return info_dict,names

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict:Dict, names:Dict, output:str)->None:
    ''' transfrom data into yoloV5 format from info_dict
        parameters:
        info_dict (dict): dictionary in this format {'bboxes': [{'class': ,
                                                    'xmin': ,
                                                    'ymin': , 
                                                    'xmax': , 
                                                    'ymax': 
                                                    }], 
                                                'filename': , 
                                                'image_size':
                                                }
        names (dict): dictionary of names {class_1:0, class_2:1, class_3: 2}
        output (str): output folder path
    '''

    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = names[b["class"]]
        except KeyError:

            print("Invalid Class. Must be one from ", names.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    # Name of the file which we have to save 
    save_file_name = os.path.join(output+"/labels", info_dict["filename"].replace("png", "txt"))
    
    # with open(save_file_name,"w")
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))
    # print("labels saved in YoloV5 format at ", output+"/labels")
    # print(print_buffer)


def yolov5_from_xml(xml_file_path:str = "None", xml_folder:str= "None",output:str="None")->Dict:
    '''
        transform XML file data from VOC to YOLOV5 Object detection format and saves all the transfromed data at given location
        
        parameters:
        xml_file_apth(str): xml file path need to transfrom from VOC XML to YoloV5 TXT format
        xml_folder(str): annotation folder path which contain all the xml files 
        output(str): location where all the transfromed data is stored

        returns:
        Dict: returns nafrom test import test_yolov5_transformmes of format of annoted classes 
    '''

    import os
    names = {}
    if output=="None":
        output = os.getcwd()
    try:
        os.mkdir(os.path.join(output, "labels"))
    except Exception as e:
        pass
        # print(e)
    if xml_file_path!="None" :
        info_dict,names = yolo_extract_info_from_xml(xml_file_path)
        convert_to_yolov5(info_dict,names,output)
    if(xml_folder!="None"):
        for root, _ ,files in os.walk(xml_folder):

            for file in files:
                
                if file[-4:]=='.xml':

                    xml_file = os.path.join(root, file)
                    info_dict,names = extract_info_from_xml(xml_file)
                    convert_to_yolov5(info_dict,names,output)
    return names



if __name__ == "__main__":
    # yolov5_from_xml(xml_folder="xml_folder")
    test_all()
