import os
import glob
import xml.etree.ElementTree as ET

def parse_dfsar_xml(xml_path):
    print(f"\n{'='*50}\nInspecting: {os.path.basename(xml_path)}\n{'='*50}")
    
    ns = {
        'pds': 'http://pds.nasa.gov/pds4/pds/v1',
        'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'
    }
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 1. Product type
    # Usually found in logical_identifier or processing_level
    logical_id = root.find('.//pds:logical_identifier', ns)
    if logical_id is not None:
        print(f"Logical Identifier: {logical_id.text}")
        
    processing_level = root.find('.//pds:processing_level', ns)
    if processing_level is not None:
        print(f"Processing Level: {processing_level.text}")
        
    # 2. Number of bands and names
    arrays = root.findall('.//pds:Array_3D_Image', ns) or root.findall('.//pds:Array_3D', ns) or root.findall('.//pds:Array_2D_Image', ns)
    print(f"\nFound {len(arrays)} array structures.")
    
    for i, arr in enumerate(arrays):
        axes = arr.findall('pds:Axis_Array', ns)
        shape = [a.find('pds:elements', ns).text for a in axes]
        ax_names = [a.find('pds:axis_name', ns).text for a in axes]
        dtype = arr.find('.//pds:data_type', ns).text
        print(f"Array {i}: Shape {' x '.join(shape)} ({' x '.join(ax_names)}), dtype={dtype}")
        
        # Band names if it's a 3D array with bands
        band_axis = None
        for ax in axes:
            if ax.find('pds:axis_name', ns).text == 'Band':
                band_axis = ax
                break
                
        if band_axis is not None:
            print("Band names mapping:")
            # ISDA namespace sometimes holds band info or it's in Local_Dictionary
            
    # Try to find specific band mapping in ISDA extension
    isda_sar = root.find('.//isda:SAR_Parameters', ns)
    if isda_sar is not None:
        print("\nSAR Parameters:")
        freq = isda_sar.find('isda:radar_center_frequency', ns)
        if freq is not None:
            print(f"Center Frequency: {freq.text} {freq.get('unit','')}")
            
        pol = isda_sar.find('isda:polarization_type', ns)
        if pol is not None:
            print(f"Polarization Type: {pol.text}")
            
    # 3. Pixel spacing
    isda_geom = root.find('.//isda:Geometry_Parameters', ns)
    if isda_geom is not None:
        print("\nGeometry / Pixel Spacing:")
        ps_az = isda_geom.find('.//isda:pixel_spacing_azimuth', ns)
        ps_rg = isda_geom.find('.//isda:pixel_spacing_range', ns)
        if ps_az is not None and ps_rg is not None:
            print(f"Pixel Spacing: Azimuth = {ps_az.text} m, Range = {ps_rg.text} m")
            
    # 4. Geographic bounds
    sys_coords = root.find('.//isda:System_Level_Coordinates', ns)
    if sys_coords is not None:
        print("\nGeographic Bounds:")
        ul_lat = sys_coords.find('isda:upper_left_latitude', ns).text
        lr_lat = sys_coords.find('isda:lower_right_latitude', ns).text
        ul_lon = sys_coords.find('isda:upper_left_longitude', ns).text
        lr_lon = sys_coords.find('isda:lower_right_longitude', ns).text
        print(f"Lat: {ul_lat} to {lr_lat}")
        print(f"Lon: {ul_lon} to {lr_lon}")


if __name__ == "__main__":
    base_dir = r"c:\Users\MRaza\Documents\Isro-BAH-RS\dfsar_data"
    xmls = glob.glob(os.path.join(base_dir, '**', '*.xml'), recursive=True)
    if not xmls:
        print("No XML files found.")
    for xml in xmls:
        parse_dfsar_xml(xml)
