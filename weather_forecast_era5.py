import cdsapi

def retrieve_weather(date):
        c = cdsapi.Client()
        year = str(date.year)
        day = str(date.day)
        month = str(date.month)

        c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'year': year,
            'month': month,
            'day': day,
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                43.4, 2.3, 43.45,
                2.35
            ],

            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', 'total_cloud_cover', '2m_temperature',
                'surface_net_solar_radiation', 'surface_net_thermal_radiation'
            ],
        },
        'meteo_file.nc')