Select *
From [Portfolio project]..[Covid Deaths]
Where continent is not null
order by 3,4
--From [Portfolio project]..[Covid Vaccinations]
--order by 3,4

-- Select Data that we are going to be using
--Select *
--From [Portfolio project]..[Covid Vaccinations]
--order by 3,4

Select location, date, new_cases, total_cases, total_deaths, population
From [Portfolio project]..[Covid Deaths]
order by 1,2

--Looking at Total Cases vs Total Deaths

--comment as of 10/11/2021 there is a 1.62% chance of dying from covid in the united states
-- Comment as of 10/11/2021 there is a 1.51% chance of dying from covid in the united kingdom

Select location, date, total_cases,total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
From [Portfolio project]..[Covid Deaths]
Where location like '%United K%'
order by 1,2

--Looking at the Total Cases vs Population
-- Shows what perecetnage of population got Covid


Select Location, date, population, total_cases, (total_cases/population)*100 as DeathPercentage
From [Portfolio project]..[Covid Deaths]
Where location like '%states%'
order by 1,2

----From data we find that as of 10/11/21 in the united states perecentage of population who had covid was 14.05%


4
Looking at countrires with the highest infection rate compared to its population

Select Location, population, MAX(total_cases) as HighestInfectionCount, MAX(total_cases/population)*100 as PercentagePopulationInfected
From [Portfolio project]..[Covid Deaths]
--Where location like '%states%'
Group by Location, Population
order by PercentagePopulationInfected desc

------ As we can see  from the this query result Montegnegro had the highest perecentage of people infected with the coronavirus
--- this was 23.9% of the population which is close to a quater. However Monetenegro has relatively small population. 
--Compared to a country like United Kingdom which had 13.85% infected with the coronovirus.

5 --Showing countries with Highest Death count per poulation 

Select Location, MAX(Cast(Total_deaths as int)) as TotalDeathCount
From [Portfolio project]..[Covid Deaths]
--Where location like '%states%'
where continent is not null
Group by Location
order by TotalDeathCount desc

--table query results shows that the  united states that the highest total dweath out of all  224 nations where data was recorded the united states has teh highest corona
--related deaths acccording the the data. USA recorded 758,916 deaths.



6 Showing continents with the highest dead count by continent

Select continent, MAX(Cast(Total_deaths as int)) as TotalDeathCount
From [Portfolio project]..[Covid Deaths]
--Where location like '%states%'
where continent is not null
Group by continent
order by TotalDeathCount desc

--comment - table above shows death toll count by continent. cna replace above queries with continent as well.


7 --Global numbers - not filterting out by location or continent by just global numbers

Select SUM(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, SUM(cast (new_deaths as int))/Sum(new_cases)*100 as DeathPercentage
 From [Portfolio project]..[Covid Deaths]
--Where location like '%states%'
where continent is not null
--Group by date
order by 1,2

--judging by the query table across the world the death perecntage is a little over 2.01%



--8 - Looking at total population vs Vaccinations

---- USE CTE

With PopvsVac (Continent, Location, Date , Population , New_Vaccinations, RollingPeopleVaccinated)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition by dea.location order by dea.location, dea.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
From [Portfolio project]..[Covid Deaths] dea
Join [Portfolio project]..[Covid Vaccinations] vac
       on dea.location = vac.location
	   and dea.date = vac.date
where dea.continent is not null
--order by 2,3
)
Select* , (RollingPeopleVaccinated/Population)*100
From PopvsVac


-- TEMP TABLE

DROP Table if exists #PercentagePopulationVaccinated
Create Table #PercentagePopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_Vaccinations numeric,
RollingPeopleVaccinated numeric
)

Insert into #PercentagePopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, 
dea.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
From [Portfolio project]..[Covid Deaths] dea
Join [Portfolio project]..[Covid Vaccinations] vac
       on dea.location = vac.location
	   and dea.date = vac.date
where dea.continent is not null
order by 2,3

Select* , (RollingPeopleVaccinated/Population)*100
From #PercentagePopulationVaccinated

---Creating view to store data for later visualisation

Create View PercentagePopulationVaccianted as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, 
  dea.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
From [Portfolio project]..[Covid Deaths] dea
Join [Portfolio project]..[Covid Vaccinations] vac
       On dea.location = vac.location
	   and dea.date = vac.date
where dea.continent is not null
--order by 2,3

Select *
From PercentagePopulationVaccianted
