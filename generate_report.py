import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from matplotlib.dates import SecondLocator
import datetime
import bs4
import lxml
import shutil
import os

# Copying template -
if os.path.isdir('ECG_Report'):
    shutil.rmtree('ECG_Report')
shutil.copytree('template', 'ECG_Report')
os.rename('ECG_Report/consolidationtemplate.html', 'ECG_Report/my_report.html')

with open("100.json") as hundred:
    data = json.load(hundred)

with open("patient_details.json") as p_details:
    patient_details = json.load(p_details)

with open("ECG_Report/my_report.html") as inf:
    txt = inf.read()
    soup = bs4.BeautifulSoup(txt, "lxml")


def soup_replace(eye_d, data_html):
    global soup
    tag = soup.find(id=eye_d)
    tag.insert(0, data_html)
    soup.find(id=eye_d).replace_with(tag)


def hr_summary(location, hr, ann, timeratio, ratio, img):
    def convert(n):
        dt = datetime.datetime.strptime('0', '%M')
        dt = dt + datetime.timedelta(seconds=n)
        return dt

    scatter_fhr = pd.DataFrame([location, hr, ann],
                               index=['time', 'bpm', 'beat_ann']).T
    scatter_fhr['time'] = (scatter_fhr['time'] * timeratio * ratio) / 1000
    scatter_fhr['time'] = scatter_fhr['time'].apply(convert)

    fig, ax = plt.subplots(1, figsize=(14, 3))
    ax.scatter(scatter_fhr[scatter_fhr['beat_ann'] == 'V']['time'],
               scatter_fhr[scatter_fhr['beat_ann'] == 'V']['bpm'],
               marker='s', label='Ventricular Ectopics', c='red', s=10)
    ax.scatter(scatter_fhr[scatter_fhr['beat_ann'] == 'A']['time'],
               scatter_fhr[scatter_fhr['beat_ann'] == 'A']['bpm'],
               marker='s', label='Atrial Ectopics', c='blue', s=10)
    ax.scatter(scatter_fhr[scatter_fhr['beat_ann'] == 'N']['time'],
               scatter_fhr[scatter_fhr['beat_ann'] == 'N']['bpm'],
               marker='s', label='Heart Rate', c='green', s=10)

    ax.set_xlim(datetime.datetime.strptime('0', '%M')
                , datetime.datetime.strptime('0', '%M') + datetime.timedelta(minutes=32))
    ax.set_ylim(0, 200)

    plt.xticks([datetime.datetime.strptime('0', '%M') + datetime.timedelta(minutes=2 * n) for n in
                range(17)],
               [(datetime.datetime.strptime('0', '%M') + datetime.timedelta(
                   minutes=2 * n)).strftime("%H:%M:%S") for n in range(17)])

    ax.set_ylabel('bpm')
    ax.set_xlabel('Time')

    ax.xaxis.set_minor_locator(SecondLocator(interval=20))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerfirst=False)
    ax.set_axisbelow(True)
    plt.grid(b=True, which='major', color='#ff0000', linestyle='-', lw=0.7)
    plt.grid(b=True, which='minor', color='#cf2d21', linestyle='-', alpha=0.5, lw=0.3)

    plt.tight_layout(pad=0.5)
    plt.savefig(img)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def stackbar_plot(plot, img):
    ls = plot
    ls1 = []
    for x in ls:
        ls1.append(x.split(" : "))
    df = pd.DataFrame(ls1, columns=['bar', 'value'])
    df['value'] = df['value'].apply(lambda u: float(u[:-1]))

    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.set_xlim(0, 7)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    plt.bar(0, df[df['bar'] == 'Sinus Rhythm']['value'], width=1, color='blue',
            label='Sinus Rhythm')

    plt.bar(0, df[df['bar'] == 'Atrial Fibrillation']['value'], width=1, color='yellow',
            bottom=float(df[df['bar'] == 'Sinus Rhythm']['value']), label='Atrial Fibrillation')

    plt.bar(0, df[df['bar'] == 'Tachycardia (>100 bpm)']['value'], width=1, color='green',
            bottom=float(df[df['bar'] == 'Sinus Rhythm']['value']) +
                   float(df[df['bar'] == 'Atrial Fibrillation']['value']),
            label='Tachycardia (>100 bpm)')

    plt.bar(0, df[df['bar'] == 'Bradycardia (<60 bpm)']['value'], width=1, color='red',
            bottom=float(df[df['bar'] == 'Sinus Rhythm']['value']) +
                   float(df[df['bar'] == 'Atrial Fibrillation']['value']) +
                   float(df[df['bar'] == 'Tachycardia (>100 bpm)']['value']),
            label='Bradycardia (<60 bpm)')

    plt.ylabel("Percentage %")
    plt.legend(loc='upper center', markerfirst=False, prop={'size': 8})

    plt.tight_layout(pad=0.5)
    plt.savefig(img)
    # plt.show()
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def plot_rate(plots, summary, location, hr, id, img):
    df = pd.DataFrame(plots)
    x_major_locator = df.shape[0] / 30
    x_minor_locator = x_major_locator / 5
    y_minor_locator = 0.1

    x = np.arange(0, len(plots))
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_np = scaler.fit_transform(df)

    fig, ax = plt.subplots(1, figsize=(15, 2))
    # fig.suptitle('Occurrence of %s Heart Rate (%s)' % (title, summary), size=10, c="#0041DC",
    #            fontweight='bold', y=0.96)
    soup_replace(id, summary)
    ax.set_ylabel('Amplitude')

    ax.plot(x, normalized_np, color='black', lw='1')

    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    ax.xaxis.set_major_formatter(FormatStrFormatter(''))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor_locator))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_locator))

    plt.xlim([0, df.shape[0]])
    plt.ylim([-1.1, 1.3])
    plt.grid(b=True, which='major', color='#ff0000', linestyle='-', lw=0.7)
    plt.grid(b=True, which='minor', color='#cf2d21', linestyle='-', alpha=0.5, lw=0.3)
    ax.set_axisbelow(True)

    for i in range(1, len(location)):
        # plt.arrow(location[i - 1], 0, (location[i] - location[i - 1]), 0, head_width=0.06,
        # head_length=7, linewidth=1.5, color='blue', length_includes_head=True)
        plt.arrow(location[i], 0, -(location[i] - location[i - 1]), 0, head_width=0.08,
                  head_length=7, linewidth=1, color='blue', length_includes_head=True)
        ax.text(((location[i - 1] + location[i]) / 2) - 20, -0.3, f'HR: {hr[i - 1]}', fontsize=10,
                color='blue')

    plt.tight_layout(pad=0.5)
    plt.savefig(img)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def plot_rhy(plots, location, noise_hr, beat_ann, p_wave_location, rhy_type, img):
    hr = [x.split("_")[2] for x in noise_hr]

    df = pd.DataFrame(plots)
    x_major_locator = df.shape[0] / 30
    x_minor_locator = x_major_locator / 5
    y_minor_locator = 0.1

    x = np.arange(0, len(plots))
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_np_min = scaler.fit_transform(df)

    fig, ax = plt.subplots(1, figsize=(15, 2))
    ax.set_ylabel('Amplitude')

    ax.plot(x, normalized_np_min, color='black', lw='1')

    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    ax.xaxis.set_major_formatter(FormatStrFormatter(''))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor_locator))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_locator))

    plt.xlim([0, len(plots)])
    plt.ylim([-1.1, 1.3])
    plt.grid(b=True, which='major', color='#ff0000', linestyle='-', lw=0.7)
    plt.grid(b=True, which='minor', color='#cf2d21', linestyle='-', alpha=0.5, lw=0.3)
    ax.set_axisbelow(True)

    for i in range(1, len(location)):
        # plt.arrow(location[i - 1], 0, (location[i] - location[i - 1]), 0, head_width=0.06,
        # head_length=7, linewidth=1.5, color='blue', length_includes_head=True)
        plt.arrow(location[i], 0, -(location[i] - location[i - 1]), 0, head_width=0.1,
                  head_length=7, linewidth=1, color='blue', length_includes_head=True)
        ax.text(((location[i - 1] + location[i]) / 2) - 20, -0.3, f'HR: {hr[i]}', fontsize=10,
                color='blue')
        if beat_ann[i] == 'N' and beat_ann[i - 1] == 'N':
            ax.text(((location[i - 1] + location[i]) / 2) - 10, 1.1, 'Reg', fontsize=9,
                    color='blue')

    for i in range(0, len(location)):
        ax.text(location[i], 1, beat_ann[i], fontsize=10, color='black')

    for i in range(0, len(p_wave_location)):
        if beat_ann[i] == 'N' and p_wave_location[i] != 0:
            ax.text(p_wave_location[i], 0.7, 'P', fontsize=10, color='red')

    plt.tight_layout(pad=0.5)
    plt.savefig(img)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    global soup
    image_name = img.split('/')[1]
    image_tag = soup.new_tag("IMG")
    image_tag['SRC'] = image_name
    image_tag['class'] = 'annotation-li'
    hr_tag = soup.new_tag("HR")
    hr_tag['NOSHADE'] = ""
    hr_tag['ALIGN'] = "CENTER"
    hr_tag['SIZE'] = "2"

    if rhy_type == "Normal Sinus Rhythm":
        soup.find(id='p-a-c').insert_before(image_tag)
        soup.find(id='p-a-c').insert_before(hr_tag)
    elif rhy_type == "PAC (Premature Atrial Contraction)":
        soup.find(id='p-v-c').insert_before(image_tag)
        soup.find(id='p-v-c').insert_before(hr_tag)
    elif rhy_type == "PVC (Premature Ventricular Contraction)":
        soup.find(id='end-of-images').insert_before(image_tag)
        soup.find(id='end-of-images').insert_before(hr_tag)


# Heart Rate Summary section (ID: 66-48) -
hr_summary(data['HRcart']['beat_location'], data['HRcart']['F_HR'], data['HRcart']['Beat_ann'],
           data['HRcart']['timeRatio'], data['HRcart']['ratio'], 'ECG_Report/HR_chart.png')

# Automated Interpretation Chart section (ID: 66-38) -
stackbar_plot(data['bargraph_details'], 'ECG_Report/bar_chart.png')
li = data['bargraph_details']
li.reverse()
soup.find("img", {"src": "bar_chart.png"})
for i in li:
    legend_list = soup.new_tag("LI")
    legend_list['style'] = "font-size: 11pt;"
    legend_list.append(i)
    soup.find("img", {"src": "bar_chart.png"}).insert_after(legend_list)

# Occurrence of Minimum Heart Rate (ID: 66-39) -
plot_rate(data['MinHR']['data'], '(' + data['Summary']['minHR'] + ')', data['MinHR']['location'],
          data['MinHR']['HR'], 'occurance-minhr', 'ECG_Report/min.png')

# Occurrence of Minimum Heart Rate (ID: 66-42) -
plot_rate(data['MaxHR']['data'], '(' + data['Summary']['maxHR'] + ')', data['MaxHR']['location'],
          data['MaxHR']['HR'], 'occurance-maxhr', 'ECG_Report/max.png')

nsr = soup.new_tag("H3")
nsr['style'] = "text-align:center; color:#0041DC"
nsr['id'] = "n-s-r"
nsr.append("Normal Sinus Rhythm")
soup.find(id='end-of-images').insert_before(nsr)

pac = soup.new_tag("H3")
pac['style'] = "text-align:center; color:#0041DC"
pac['id'] = "p-a-c"
pac.append("PAC (Premature Atrial Contraction)")
soup.find(id='end-of-images').insert_before(pac)

pvc = soup.new_tag("H3")
pvc['style'] = "text-align:center; color:#0041DC"
pvc['id'] = "p-v-c"
pvc.append("PVC (Premature Ventricular Contraction)")
soup.find(id='end-of-images').insert_before(pvc)

#
for i in range(0, len(data['Rhy_details'])):
    for j in range(0, len(data['Rhy_details'][i])):
        for k in range(0, len(data['Rhy_details'][i][j]['data'])):
            if data['Rhy_details'][i][j]['RHY'] != "" and (k < 3):  # each rhythm 3 samples
                plot_rhy(data['Rhy_details'][i][j]['data'][k]['data'],
                         data['Rhy_details'][i][j]['data'][k]['location'],
                         data['Rhy_details'][i][j]['data'][k]['Noise_HR_TAG'],
                         data['Rhy_details'][i][j]['data'][k]['beat_ann'],
                         data['Rhy_details'][i][j]['data'][k]['p_wave_location'],
                         data['Rhy_details'][i][j]['RHY'],
                         f'ECG_Report/annotation{i}{j}{k}.png')

# Report Generation Date (ID: 66-37) (point 3) -
cut_time = datetime.datetime.utcnow()
cut_time = f'{cut_time.strftime("%m/%d/%Y %H:%M:%S")} CUT'
soup_replace("timedate", cut_time)

# ECG REPORT section (ID: 66-45) -
soup_replace("report-number", patient_details['ReportNumber'])
soup_replace("patient-id", patient_details['patient_details']['medicalRecordNumber'])
soup_replace("physician", patient_details['patient_details']['referredBy'])
soup_replace("dob", patient_details['patient_details']['dob'])
soup_replace("device-id", patient_details['patient_details']['deviceId'])
soup_replace("age", str(patient_details['patient_details']['age']) + ' years')
soup_replace("start-date", patient_details['patient_details']['startOfRecording'])
soup_replace("gender", patient_details['patient_details']['gender'])
soup_replace("end-date", patient_details['patient_details']['endOfRecording'])
soup_replace("primary-indication", patient_details['patient_details']['primaryIndication'])
sor = datetime.datetime.strptime(patient_details['patient_details']['startOfRecording'],
                                 '%m/%d/%Y %H:%M:%S')
eor = datetime.datetime.strptime(patient_details['patient_details']['endOfRecording'],
                                 '%m/%d/%Y %H:%M:%S')
soup_replace("record-time", str(eor - sor))
soup_replace("medications", patient_details['patient_details']['medications'])
soup_replace("medications", patient_details['patient_details']['medications'])
soup_replace("analyzed-time", data['HRcart']['Time'])

# General section (ID: 66-46) -
soup_replace("rhythm", data['general']['RHYTHM'])
soup_replace("qrs", data['general']['QRS '])
soup_replace("p-interval", data['general']['PWave'])
soup_replace("pr-interval", data['general']['PRI'])
soup_replace("predicted-beats", data['general']['OTHER'])
soup_replace("qt", data['general']['QT'])
soup_replace("qtc", data['general']['QTc'])
soup_replace("percentage-noise", data['general']['noise'])

# Summary section (ID: 66-43) -
soup_replace("minimum-hr", data['Summary']['minHR'])
soup_replace("average-hr", data['Summary']['avgHR '])
soup_replace("maximum-hr", data['Summary']['maxHR'])
soup_replace("total-beats", data['Summary']['totalbeatcount'])
ve = f'%.2f' % ((data['HRcart']['Beat_ann'].count('V') / len(data['HRcart']['Beat_ann'])) * 100)
se = f'%.2f' % ((data['HRcart']['Beat_ann'].count('A') / len(data['HRcart']['Beat_ann'])) * 100)
soup_replace("supraventricular", se + '%')
soup_replace("ventricular", ve + '%')
soup_replace("supra-isolated", str(data['Summary']['pvc']['Isolated']))
soup_replace("supra-total", str(data['Summary']['pvc']['Total']))
soup_replace("ventri-isolated", str(data['Summary']['pac']['Isolated']))
soup_replace("ventri-total", str(data['Summary']['pac']['Total']))

# Arrhythmia Data section (ID: 66-44) -
soup_replace("af-beats", data['ArrhythmiaData']['AFBurden'])
soup_replace("vt", data['ArrhythmiaData']['vt'])
soup_replace("pauses", data['ArrhythmiaData']['Pauses'])
soup_replace("svt", data['ArrhythmiaData']['svta'])

# print(soup)
soup = soup.prettify()
with open("ECG_Report/my_report.html", "w") as outf:
    outf.write(str(soup))
