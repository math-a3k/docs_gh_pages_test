# -*- coding: utf-8 -*-
from ua_parser import user_agent_parser

uaParseCommon = user_agent_parser.Parse


def getall_families_from_useragent(ua_string):
    if (ua_string != ua_string) | (ua_string is None):
       return ' - - '
    
    all_families = uaParseCommon(ua_string)
    all3_families = all_families.get('os').get('family') + "-" + all_families.get('device').get('family') + "-" + all_families.get('user_agent').get(
        'family')
    return all3_families







