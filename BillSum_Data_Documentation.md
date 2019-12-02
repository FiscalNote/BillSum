For each bill, there are two necessary files: a metadata file (that includes the summary) and the file with the text. 

We store the final version of the data [here](https://drive.google.com/file/d/1g89WgFHMRbr4QrvA0ngh26PY081Nv3lx/view?usp=sharing), but document the full procedure for reference. First we document how to extract the source files, then how to create the final dataset.

---
# Source data

The procedure is split into two parts, because Congress changed the format/location that the data is stored in the 113th session.

## 103-112 session 

The procedure is adapted from GovTrack's [public domain code](https://github.com/unitedstates/congress) for collecting bill data. This is an excellent resource for gathering other Congress related data.

**Metadata**: Download [this archive](http://thomas.gov/). It is a copy of a deprecated resource from the library of congress. The data in this archive is stored in the following format:

```
- session (e.g 104)
    - bills
        - bill type (e.g h, hres)
            - bill num
                - data.{json, xml}
```

The data files contain the same information, just in the respective formats.

**Text**: 

Extracted using the `congress` library above:

1. Clone `https://github.com/unitedstates/congress`
2. Install the python dependecies
3. Modify the `tasks/utils.py` file to change line 515 to point to the root of the Thomas.gov repo. (If you extracted in Thomas.gov to `/MY/PATH/data` - change the line to that path)
4. Run `./run govinfo --collections=BILLS --congress=110 --store=text,xml --extract=text,xml` for every congress from 103-112.

Note: This procedure takes several hours.


The text files will be stored in the following format:
```
- session (e.g 105)
    - bills
        - bill type (e.g h, hres)
            - bill num
                - text-versions
                    - bill version 
                        - document.{html, xml}
```

Note: xml is only available for a subset of the 111,112 bills.

## 113-115 sessions

To get the texts, we repeat the command from above:

`./run govinfo --collections=BILLS --congress=114 --store=text,xml --extract=text,xml`

To get the metadata, run the `gpo_scraper.py` script in this repo. Change the local path to be the root of your data directory. The script will scrape the data from `https://www.govinfo.gov/bulkdata/`. The final files will be the same as the data.xml files in the above structure.


# Creating final dataset 

For the final dataset, we extract the summary + final version of the text. We choose this pair, because earlier versions of the summary are not always available and because it is difficult to match up earlier texts to summaries.

The `prepare_bills.py` script will take care of matching bill texts to summaries, extracting the text from xml creating final records of titles/summaries/texts.

