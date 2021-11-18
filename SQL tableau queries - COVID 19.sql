--1. table 1 - Global numbers - Tableau Visualisation


--This table is designed to showcase the sum of new covid cases in the world, this will be a new columm called total cases .The sum
--of new covid deaths will be a new columm called total deaths. Dividing the total deaths by the total cases will give a death percentage. 
--Results show there is a 2.01% of dying from covid as of Novemebr 2021.


Select SUM(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, SUM(cast (new_deaths as int))/Sum(new_cases)*100 as DeathPercentage
 From [Portfolio project]..[Covid Deaths]

where continent is not null

order by 1,2



--table 2 - CoronaVirus death count by continent

--We take out these as they are not included in the above SQL queries and we want to keep consistency with our data. European union is part of europe, we also want to take
--out 'upper income', 'low income', 'lower middle class', 'high income', 'international', 'world' as they are not continents.

Select location, SUM(cast(new_deaths as int)) as totaldeathcount
From [Portfolio project]..[Covid Deaths]
--Where location like '%states%'
Where continent is null
and location not in('World', 'European Union', 'International', 'High Income','low income','upper middle income','lower middle income')
Group by location
order by totaldeathcount desc



--table 3 - - Geographical map - You can type in the search box a country and see its infection rates

Select Location, population, MAX(total_cases) as HighestInfectionCount, MAX(total_cases/population)*100 as PercentagePopulationInfected
From [Portfolio project]..[Covid Deaths]
--Where location like '%states%'
Group by Location, Population
order by PercentagePopulationInfected desc


--table 4 - Line grpah of perecentage of population infected by covid and a future forecast indicator. 5 countries selected in graph UK, CHINA, USA, MEXCICO


Select Location, population,date, MAX(total_cases) as HighestInfectionCount, MAX(total_cases/population)*100 as PercentagePopulationInfected
From [Portfolio project]..[Covid Deaths]
--Where location like '%states%'
Group by Location, Population, date
order by PercentagePopulationInfected desc
