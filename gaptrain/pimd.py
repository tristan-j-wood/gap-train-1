import xml.etree.cElementTree as ET
from xml.dom import minidom


class PIMD:
    """Path integral molecular dynamics class to run PMID simulations using
       the i-PI Python interface."""

    def write_xml_file(self, output_name='output', md_stride="1"):
        """Writes the xml input file required for i-PI"""

        property_list = "[step, time{picosecond}, conserved{kelvin}, " \
                        "temperature{kelvin}, potential{kelvin}, " \
                        "kinetic_cv{kelvin}]"

        root = ET.Element("Simulation", verbosity='high')

        # Output block
        output = ET.SubElement(root, "output", prefix=output_name)
        ET.SubElement(output, "properties", filename="md", stride=md_stride)

        # Write file with indentation
        xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(
            indent="   ")
        with open("input.xml", "w") as f:
            f.write(xml_string)

        return None

    def run_pimd(self):

        return NotImplementedError

    def __init__(self):

        self.test = 'test'
