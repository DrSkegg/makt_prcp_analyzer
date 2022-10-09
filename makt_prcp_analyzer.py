#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from scipy.interpolate import interp1d
from glob import glob
import os.path as op
import os
from tqdm import tqdm
from datetime import datetime
import platform
import configparser
import numba
from collections import OrderedDict



def groupping_quartal(data):
    m12_index = (data.index.month == 12)
    corrected_years = pd.Series(data.index.year, index=data.index)
    corrected_years[m12_index] += 1
    corrected_months = pd.Series(data.index.month, index=data.index)
    corrected_months[m12_index] = 0
    corrected_months //= 3
    corrected_months += 1
    groupped_by_quartal = data.groupby([corrected_years, corrected_months])
    return groupped_by_quartal


def empf(x, empr, kind="linear"):
    x = np.array(x)
    maxval = empr[:, 0].max()
    x[x >= maxval] = maxval
    x[x < 0.] = np.nan
    return np.round(interp1d(empr[:, 0], empr[:, 1], kind=kind)(x), 4)


@numba.jit(nopython=True)
def calculate_periods(x, duration=5):
    periods_num = 0
    periods_common_len = 0
    q = 0
    for i in x:
        if i == 0:
            if q >= duration:
                periods_common_len += q
            q = 0
        else:
            q += 1
        if q == duration:
            periods_num += 1

    if q >= duration:
        periods_common_len += q

    return (periods_num, periods_common_len)


@numba.jit(nopython=True)
def analyze_periods(x, rang90, rang95, gamma90, gamma95, nanthreshold=0.1):
    #  NDRGTP90:   число дней с осадками выше 90 процентиля;
    #  NDRGTP95:    число дней с осадками выше 95 процентиля;
    #  NDRGT10:  числа дней с осадками выше 10 мм
    #  NDREQ10: числа дней без дождя;
    #  NS5RGTP90, NDS5RGTP95: число и суммарная продолжительность дождливых периодов: от 5 дней выше 90 процентиля;
    #  NDS5REQ0,  NDS5REQ0: число и суммарная продолжительность бездождных периодов от 5 дней

    nanx = np.isnan(x)
    nanmean = nanx.mean()
    # if x.size < 89 or nanmean > nanthreshold:
    #     return (np.round(nanmean, 4), -9999) + (-9999,) * 12
    if x.size < 89:
        return (np.round((90 - x.size +nanx.sum())/90., 4), -9999) + (-9999,) * 12
    if nanmean > nanthreshold:
        return (np.round(nanmean, 4), -9999) + (-9999,) * 12

    # Заполняем отсуствующие данные средним значением
    x[nanx] = np.nanmean(x)

    prcp_gt_rang90 = np.greater(x, rang90)
    prcp_gt_gamma90 = np.greater(x, gamma90)

    NDRGTP90 = prcp_gt_rang90.sum()
    NDRGTG90 = prcp_gt_gamma90.sum()

    NDRGTP95 = np.greater(x, rang95).sum()
    NDRGTG95 = np.greater(x, gamma95).sum()
    NDRGT10 = np.greater(x, 10.).sum()
    prcp_eq_0 = np.equal(x, 0.)
    NDREQ10 = prcp_eq_0.sum()

    NS5RGTP90, NDS5RGTP95 = calculate_periods(prcp_gt_rang90)
    NS5RGTG90, NDS5RGTG95 = calculate_periods(prcp_gt_gamma90)

    NS5REQ0, NDS5REQ0 = calculate_periods(prcp_eq_0)

    SUMR = np.round(x.sum(), 4)

    return (np.round(nanmean, 4), SUMR,
            NDRGT10, NDREQ10,
            NS5REQ0, NDS5REQ0,
            NDRGTP90, NDRGTP95,
            NS5RGTP90, NDS5RGTP95,
            NDRGTG90, NDRGTG95,
            NS5RGTG90, NDS5RGTG95,
            )


def spi(x, a, b, q):
    '''
    Функция рассчета SPI по заданным значенияем параметров гамма-распределения и q
    '''
    x = x.copy()
    x[x < 0.] = np.nan
    G = q + (1. - q) * gamma.cdf(x, loc=0., a=a, scale=b)
    spi_ = norm.ppf(G)

    spi_[np.isnan(spi_)] = -9999.

    return spi_


class MaktPrcpExtractor:
    def __init__(self, config):

        self.starttime = datetime.now()
        self.tolog = False
        self.prcp_from_makt = None

        print(self.starttime.strftime("%Y-%m-%d %H:%M:%S"))

        try:
            self.stations_list_file = config["DEFAULT"]["STATIONS_LIST_FILE"]
            self.makt_dir = config["DEFAULT"]["MAKT_FILES_DIR"]
            self.output_dir = config["DEFAULT"]["OUTPUT_DIR"].format(now=self.starttime.strftime("%Y-%m-%d_%H:%M:%S"))
            self.upper_prcp_threshold = float(config["DEFAULT"]["UPPER_PRCP_THRESHOLD"]) * 100.
            self.omitted_data_threshold = float(config["DEFAULT"]["OMITTED_DATA_THRESHOLD"])
        except Exception:
            print("Конфигурационный файл имеет неправильную структуру")
            if platform.system() == "Windows":
                input("Press ENTER to exit")
            sys.exit(1)

        self.stations_list = {x.strip() for x in open(self.stations_list_file).readlines()}
        self.processed_makt_files = []
        self.failed_makt_files = []
        self.prcp_output_file = op.join(self.output_dir, "prcp.csv")
        self.spi_params_1m = pd.read_csv("precalculated/spi_gamma_params_1mon.csv", dtype={"WMO": str}).set_index("WMO")
        self.spi_params_3m = pd.read_csv("precalculated/spi_gamma_params_3mon.csv", dtype={"WMO": str}).set_index("WMO")
        self.seasonal_mean = pd.read_csv("precalculated/mean_season_prcp_sums.csv", index_col="SEASON")

        self.makt_read_ok = False
        self.test_source = False
        os.makedirs(self.output_dir, exist_ok=True)
        self.tolog = True
        # self.tolog = True

    def read_makt(self):
        print("\nЧитаем файлы MAKT")

        files_list = sorted(x for x in glob(op.join(self.makt_dir, "**"), recursive=True) if not op.isdir(x))
        self.processed_makt_files = []
        self.failed_makt_files = []

        prcp_from_makt = []

        for file in tqdm(files_list[:]):
            try:
                data = {"DATE": [], "WMO_INDEX": [], "PRCP_MAKT": []}
                for line in open(file).readlines():
                    try:
                        WMO_INDEX = line[9:14].replace(" ", "0")
                        if not WMO_INDEX in self.stations_list: continue

                        PRCP_MAKT = np.float32(line[188:195])
                        if PRCP_MAKT == -9999.: continue

                    except ValueError:
                        continue

                    data["WMO_INDEX"].append(WMO_INDEX)
                    data["DATE"].append(line[0:8].replace(" ", "0"))
                    data["PRCP_MAKT"].append(PRCP_MAKT)
                data = pd.DataFrame(data)
                data["DATE"] = pd.to_datetime(data["DATE"], format="%Y%m%d")

                data.set_index(["DATE", "WMO_INDEX"], inplace=True)
                # Фильтрация
                data[(data < 0.) | (data > self.upper_prcp_threshold)] = np.nan

                # Преобразование данных
                index999 = data["PRCP_MAKT"].between(9990., 9999.)
                data[index999] = data[index999].sub(9990.).div(10.).round(1)
                data[~index999] = data[~index999].div(100.).round(1)

                data = data.unstack(level=1, fill_value=np.nan)
                data.columns = data.columns.levels[1].set_names("")
                prcp_from_makt.append(data)
                self.processed_makt_files.append(file)


            except:
                self.failed_makt_files.append(file)

        if len(self.failed_makt_files) > 0:
            print("Не удалось прочитать файлы:")
            for i in self.failed_makt_files: print("  ", i)

        if len(prcp_from_makt) == 0:
            print("Нет валидных файлов MAKT")
            if platform.system() == "Windows":
                input("Press ENTER to exit")
            sys.exit(1)

        prcp_from_makt = pd.concat(prcp_from_makt, axis=0, sort=False).sort_index()
        prcp_from_makt = prcp_from_makt[sorted(prcp_from_makt.columns)]

        # Убираем дупликаты по дате
        nondupindex = ~prcp_from_makt.index.duplicated(keep=False) | prcp_from_makt.index.duplicated(keep="first")
        prcp_from_makt = prcp_from_makt[nondupindex]

        # Приводим к непрерывному временному ряду
        try:
            general_index = pd.date_range(start=prcp_from_makt.index.min(), end=prcp_from_makt.index.max(), freq="D")
            empty_data = pd.DataFrame(index=general_index)
            prcp_from_makt = pd.concat([empty_data, prcp_from_makt], axis=1)
            prcp_from_makt.index.name = "DATE"
        except:
            print("Что-то не так в файлах МАКТ")
            if platform.system() == "Windows":
                input("Press ENTER to exit")
            sys.exit(1)

        print(f"Сохраняем экстрагированные данные об осадках в файл {self.prcp_output_file}...",
              end="")
        self.prcp_from_makt = prcp_from_makt.loc[
            np.logical_or(prcp_from_makt.index.month != 2, prcp_from_makt.index.day != 29)]

        outdata = pd.concat (
            [
                prcp_from_makt.index.year.to_series(index=prcp_from_makt.index, name="YEAR"),
                prcp_from_makt.index.month.to_series(index=prcp_from_makt.index, name="MONTH"),
                prcp_from_makt.index.day.to_series(index=prcp_from_makt.index, name="DAY"),
                prcp_from_makt.fillna (-1.)
            ],
        axis=1).values

        FMT = ("%4d%6d%4d" + "%6.1f" * len(prcp_from_makt.columns))
        HEADER = "YEAR MONTH DAY " + " ".join(prcp_from_makt.columns)
        np.savetxt(self.prcp_output_file, outdata, fmt=FMT, header=HEADER, comments="")
            # to_csv(self.prcp_output_file)
        print(" OK")
        # prcp_from_makt.to_csv(self.prcp_output_file)
        self.makt_read_ok = True
        self.test_source = False
        return prcp_from_makt

    def add_precalculated_data(self):
        '''
        Метод считывает и добавляет к данным по станцияем заранее рассчитанные значения интервалов
        :return:
        '''
        prcp_data = self.prcp_from_makt.copy()
        prcp_data["month"] = prcp_data.index.month
        prcp_data["day"] = prcp_data.index.day
        prcp_index = prcp_data.index
        self.prcp_by_wmo = {}

        for wmo_index in self.prcp_from_makt.columns:
            percentiles = pd.read_csv(f"precalculated/percentiles/{wmo_index}.csv").iloc[:, :-2]
            prcp_wmo = prcp_data[[wmo_index, "month", "day"]].rename(columns={wmo_index: "prcp"})

            prcp_wmo = prcp_wmo.merge(percentiles)
            prcp_wmo.set_index(prcp_index, inplace=True)

            self.prcp_by_wmo[wmo_index] = prcp_wmo

    def calculate_for_seasons(self):
        climindices = ('YEAR', 'SEASON', 'nodata_fraction', 'SUMR',
                       'RSUMR',
                       'EMPR',
                       'NDRGT10',
                       'NDREQ0',
                       'NS5REQ0',
                       'NDS5REQ0',
                       'NDRGTP90',
                       'NDRGTP95',
                       'NS5RGTP90',
                       'NDS5RGTP95',
                       'NDRGTG90',
                       'NDRGTG95',
                       'NS5RGTG90',
                       'NDS5RGTG95',
                       'SPI')
        seasonal_res_all = OrderedDict()



        seasonal_res_output_dir_by_station = op.join(self.output_dir, "seasonal_results", "by_station")
        os.makedirs(seasonal_res_output_dir_by_station, exist_ok=True)


        # seasonal_res_by_index = {x:[] for x in climindices[2:]}
        header_bystation = ("%4s%7s%16s" + "%11s" * (len(climindices) - 3)) % climindices
        print("\nРассчитываем и сохраняем сезонные результаты")

        self.seasonal_res_values = []
        self.seasonal_wmo = []



        build_fmt  = True

        for wmo in tqdm(list(self.prcp_by_wmo.keys())[:]):
            gr = groupping_quartal(self.prcp_by_wmo[wmo])

            seasonal_res = OrderedDict((x, []) for x in climindices)


            for (year, season), c in gr:
                nanmean, sum, res1, res2, res3, res4, \
                res5, res6, res7, res8, \
                res9, res10, res11, res12 = analyze_periods(c["prcp"].values,
                                                            c["rang90"].values, c["rang95"].values,
                                                            c["gamma_moments_90"].values,
                                                            c['gamma_moments_95'].values,
                                                            self.omitted_data_threshold)
                seasonal_res["YEAR"].append(year)
                seasonal_res["SEASON"].append(season)
                seasonal_res["nodata_fraction"].append(nanmean)

                seasonal_res["SUMR"].append(sum)
                if sum >= 0:
                    rsumr = round(sum / self.seasonal_mean.loc[season, wmo], 4)
                else:
                    rsumr = sum
                seasonal_res["RSUMR"].append(rsumr)
                seasonal_res["NDRGT10"].append(res1)
                seasonal_res["NDREQ0"].append(res2)
                seasonal_res["NS5REQ0"].append(res3)
                seasonal_res["NDS5REQ0"].append(res4)
                seasonal_res["NDRGTP90"].append(res5)
                seasonal_res["NDRGTP95"].append(res6)
                seasonal_res["NS5RGTP90"].append(res7)
                seasonal_res["NDS5RGTP95"].append(res8)
                seasonal_res["NDRGTG90"].append(res9)
                seasonal_res["NDRGTG95"].append(res10)
                seasonal_res["NS5RGTG90"].append(res11)
                seasonal_res["NDS5RGTG95"].append(res12)

            em = np.load(op.join("precalculated/empr_seasonal", wmo + ".npy"))

            seasonal_res["EMPR"] = empf(seasonal_res["SUMR"], em)

            a, b, q = self.spi_params_3m.loc[wmo]
            spi_val = np.round(spi(np.array(seasonal_res["SUMR"]), a, b, q), 4)
            seasonal_res["SPI"] = spi_val


            seasonal_res = pd.DataFrame(seasonal_res)


            seasonal_res.fillna(-9999., inplace=True)
            # Сохраняем результаты сезонных рассчетов для станции
            seasonal_res_all[wmo] = seasonal_res
            # seasonal_res.to_csv(op.join(seasonal_res_output_dir_by_station, wmo + ".txt"), index=False, sep=" ")

            if build_fmt:
                fmt_bystation = "%4d%7d%16.4f%11.1f"
                for i in seasonal_res.dtypes[4:]:
                    if i == np.int64:
                        fmt_bystation = fmt_bystation + "%11d"
                    elif i == np.float64:
                        fmt_bystation = fmt_bystation + "%11.4f"
                build_fmt = False

            np.savetxt (op.join(seasonal_res_output_dir_by_station, wmo + ".txt"),
                        seasonal_res.values,
                        header=header_bystation,
                        fmt=fmt_bystation,
                        comments=''
                        )

            seasonal_res.set_index(["YEAR", "SEASON"], inplace=True)
            # for index in climindices[2:]:
            #   seasonal_res_by_index[index].append( seasonal_res.loc[:, [index]].rename(columns={index: wmo}))

            self.seasonal_res_values.append(seasonal_res)
            self.seasonal_wmo.append(wmo)

        #

        # seasonal_res_output_dir_by_index = op.join(self.output_dir, "seasonal_results", "by_index")
        # os.makedirs(seasonal_res_output_dir_by_index, exist_ok=True)
        #
        #
        #
        # print ("Сохранение данных по индексам...", end='')
        # for index, data in seasonal_res_by_index.items():
        #     data = pd.concat(data, axis=1).reset_index()
        #     if data.iloc[:, -1].dtype == np.int64:
        #         header = ("%4s%8s" + "%6s" * (len(data.columns) - 2)) % tuple(data.columns)
        #         fmt_bystation = "%4d%8d" + "%6d" * (data.shape[1] - 2)
        #     else:
        #         header = ("%4s%8s" + "%12s" * (len(data.columns) - 2)) % tuple(data.columns)
        #         fmt_bystation = "%4d%8d" + "%12.4f" * (data.shape[1] - 2)
        #
        #     np.savetxt (op.join(seasonal_res_output_dir_by_index, index + ".txt"),
        #                 data.values,
        #                 fmt = fmt_bystation,
        #                 header=header,
        #                 comments='')
        #         # .to_csv(op.join(seasonal_res_output_dir_by_index, index + ".txt"))
        # print (" OK")

        print("Сохранение данных по индексам...", end='')
        # joblib.dump(seasonal_res_all, op.join(self.output_dir, "seasonal_res"))
        seasonal_res_output_dir_by_index = op.join(self.output_dir, "seasonal_results", "by_index")
        os.makedirs(seasonal_res_output_dir_by_index, exist_ok=True)
        seasonal_results = []

        for wmo, results in seasonal_res_all.items():
            # print (wmo, res)
            idx = results.index.to_frame()
            idx ["WMO"] = wmo
            seasonal_results.append(results.set_index(pd.MultiIndex.from_frame(idx)))
            # print (idx)


        seasonal_results = pd.concat (seasonal_results)
        for param in seasonal_results.columns:
            data = seasonal_results[[param]].unstack().reset_index()
            if param == "SUMR":
                header = "YEAR  SEASON" + ("%8s" * (len(data.columns) - 2)) % tuple(x[1] for x in data.columns[2:])
                fmt_bystation = "%4d%8d" + "%8.1f" * (data.shape[1] - 2)
            else:
                if data.iloc[:, -1].dtype in (np.int64, np.int32):
                    header = "YEAR  SEASON" + ("%6s" * (len(data.columns) - 2)) % tuple(x[1] for x in data.columns[2:])
                    fmt_bystation = "%4d%8d" + "%6d" * (data.shape[1] - 2)

                else:
                    header = "YEAR  SEASON" + ("%11s" * (len(data.columns) - 2)) % tuple(x[1] for x in data.columns[2:])
                    fmt_bystation = "%4d%8d" + "%11.4f" * (data.shape[1] - 2)

            np.savetxt (op.join(seasonal_res_output_dir_by_index, param + ".txt"),
                        data.values,
                        fmt = fmt_bystation,
                        header=header,
                        comments='')

        print(" OK")
        # return seasonal_res_all

    def calculate_for_month(self):
        print("\nРассчитываем и сохраняем месячные результаты")
        header_station = "YEAR MONTH    SUMR       EMPR        SPI"
        fmt_station = "%4d%6d%8.1f%11.4f%11.4f"

        output_dir_station = op.join(self.output_dir, "monthly_results", "by_station")
        os.makedirs(output_dir_station, exist_ok=True)

        monthly_res = []

        for wmo in tqdm(self.prcp_from_makt.columns):
            prcp = self.prcp_from_makt[[wmo]]

            gr = prcp.groupby([prcp.index.year, prcp.index.month])

            # prcp_sum = gr.agg(lambda x: x.mean() * x.size
            #     if (x.isna().mean() < self.omitted_data_threshold) and (x.size >= 28) else -9999.)

            prcp_sum = gr.agg(lambda x: calculate_month_sum(x.values, self.omitted_data_threshold))


            prcp_sum.index.set_names(["YEAR", "MONTH"], inplace=True)
            prcp_sum.reset_index(inplace=True)
            prcp_sum.rename(columns={wmo: "SUMR"}, inplace=True)
            # prcp_sum["YEAR"] = prcp_sum.index.get_level_values(0)
            # prcp_sum["MONTH"] = prcp_sum.index.get_level_values(1)

            empr_monthly = np.load(op.join("precalculated", "empr_monthly", wmo +".npy"))

            prcp_sum["EMPR"] = empf(prcp_sum["SUMR"], empr_monthly)

            a, b, q = self.spi_params_1m.loc[wmo]
            spi_val = spi(prcp_sum["SUMR"].values, a, b, q)
            # print (spi_val)
            prcp_sum["SPI"] = spi_val
            prcp_sum.fillna(-9999., inplace=True)




            # prcp_sum.to_csv(op.join(output_dir, wmo + ".txt"), float_format="%.4f")
            np.savetxt (op.join(output_dir_station, wmo + ".txt"),
                        prcp_sum[["YEAR", "MONTH", "SUMR", "EMPR", "SPI"]].values,
                        fmt=fmt_station,
                        header=header_station,
                        comments='')

            idx = pd.MultiIndex.from_arrays([prcp_sum["YEAR"], prcp_sum["MONTH"], [wmo]*prcp_sum.shape[0]],
                                            names=["YEAR", "MONTH", "WMO"])

            monthly_res.append(prcp_sum.set_index(idx))

        monthly_res = pd.concat(monthly_res, axis=0)

        #TEST!!!
        # monthly_res.to_csv(op.join(self.output_dir, "monthly_res.csv"))

        output_dir_param = op.join(self.output_dir, "monthly_results", "by_index")
        os.makedirs(output_dir_param, exist_ok=True)



        for param in monthly_res.columns[2:]:
            param_res = monthly_res[param].unstack().reset_index()

            if param == "SUMR":
                header_param = "YEAR MONTH   " + "   ".join(param_res.columns[2:])
                fmt_param = "%4d%6d" + "%8.1f" * (len(param_res.columns) - 2)
            else:
                header_param = "YEAR MONTH      " + "      ".join(param_res.columns[2:])
                fmt_param = "%4d%6d" + "%11.4f" * (len(param_res.columns) - 2)

            np.savetxt (op.join(output_dir_param, param + ".txt"),
                        param_res.values,
                        header=header_param,
                        fmt=fmt_param,
                        comments="")


    def test_read_prcp_from_file(self):
        '''
        Тестовый метод. Читает готовые данные из файла с экстрагированными осадками
        '''
        self.test_source = True
        self.prcp_from_makt = pd.read_csv("precalculated/prcp.csv", parse_dates=["DATE"]).set_index("DATE")
        return self.prcp_from_makt

    def logging(self):
        with open(op.join(self.output_dir, "report.txt"), 'w') as f:
            print(f"Директория с выходными результатами: \"{self.output_dir}\"", file=f)
            if self.test_source:
                print("Данные об осадках взяты из готового файла", file=f)
            else:
                print(f"Директория с входящими данными: \"{self.makt_dir}\"", file=f)
                if not self.makt_read_ok:
                    print("\nВалидных файлов MAKT нет", file=f)
                else:
                    print("\nПрочитаны валидные файлы МАКТ:", file=f)
                    for i in self.processed_makt_files:
                        print(i, file=f)

                if len(self.failed_makt_files) > 0:
                    print("\nНе удалось прочитать файлы:", file=f)
                    for i in self.failed_makt_files:
                        print(i, file=f)
            if not self.prcp_from_makt is None:
                print(f"\nПроанализированы данные по {len(self.prcp_from_makt.columns)} cтанциям", file=f)
                print(f"Временной интервал данных: {self.prcp_from_makt.index.min().strftime('%Y-%m-%d')}"
                      f" -> {self.prcp_from_makt.index.max().strftime('%Y-%m-%d')}", file=f)

        print("Файл отчета сохранен")

    def __del__(self):
        if self.tolog: self.logging()


@numba.jit(nopython=True)
def calculate_month_sum (x, omitted_data_threshold):
    if (np.isnan(x).mean() < omitted_data_threshold) and (x.size >= 28):
        return np.nanmean(x) * x.size
    else:
        return -9999.


def main():
    if len(sys.argv) > 1:
        configfile = sys.argv[1]
    else:
        configfile = "config.ini"

    if not op.exists(configfile):
        print(f"Нет конфигурационного файла {configfile}")
        if platform.system() == "Windows":
            input("Press ENTER to exit")
        sys.exit(1)

    config = configparser.ConfigParser()
    print(f"Файл конфигурации {configfile}")
    try:
        config.read(configfile, encoding='utf-8')
    except Exception:
        print(f"Конфигурационный файл {configfile} имеет неправильную структуру")
        if platform.system() == "Windows":
            input("Press ENTER to exit")
        sys.exit(1)

    for k in config["DEFAULT"]:
        print(k, "  :  ", config["DEFAULT"][k])

    reader = MaktPrcpExtractor(config)
    reader.read_makt()
    # reader.test_read_prcp_from_file()
    reader.add_precalculated_data()

    with np.errstate(invalid='ignore'):
        reader.calculate_for_seasons()
        reader.calculate_for_month()

    if platform.system() == "Windows":
        input("Press ENTER to exit")


if __name__ == "__main__":
    main()
